# train_gru.py
import os
import sys
import numpy as np

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "seu_projeto.settings")  # AJUSTE
import django
django.setup()

from datetime import date, datetime
from django.utils.timezone import make_aware, utc

from core.models import PVPlant, PVPlantMergedRecord15m
from core.services.power_model import (
    expected_and_mismatch,
    module_from_pvmodule,
    plant_from_details,
    transpose_ghi_to_poa_isotropic,
)

import core.services.gru_model as gm


def fetch_clean_days(plant_id: int, start: date, end: date, source_meteo: str, interval_min: int = 15):
    """
    Retorna uma lista de dias, cada item contém arrays 96pt:
      times_utc, ghi, gti, dni, dhi, temp_air
    Critério "limpo": sem flag_meteo_missing no dia e com >= 90% pontos.
    """
    plant = PVPlant.objects.get(id=plant_id)
    tz_name = getattr(plant, "timezone", "UTC") or "UTC"

    # (você já tem _local_dates_to_utc_range na base; aqui simplifico assumindo UTC puro)
    dt0 = make_aware(datetime.combine(start, datetime.min.time()), timezone=utc)
    dt1 = make_aware(datetime.combine(end, datetime.min.time()), timezone=utc)

    qs = (
        PVPlantMergedRecord15m.objects.filter(
            plant=plant,
            source_meteo=source_meteo,
            interval_min=interval_min,
            ts_utc__gte=dt0,
            ts_utc__lt=dt1,
        )
        .order_by("ts_utc")
        .values("ts_utc", "ghi", "gti", "dni", "dhi", "temp_air", "flag_meteo_missing")
    )

    rows = list(qs)
    if not rows:
        return []

    # agrupa por dia UTC (se preferir local, adapte)
    by_day = {}
    for r in rows:
        t = r["ts_utc"]
        d = t.date()
        by_day.setdefault(d, []).append(r)

    out = []
    for d, rr in by_day.items():
        if len(rr) < 86:  # ~90% de 96
            continue
        if any(bool(x.get("flag_meteo_missing")) for x in rr):
            continue

        # garante 96 pontos (se seu banco tiver lacunas, você pode reamostrar)
        rr = rr[:96]

        out.append({
            "date": d,
            "times_utc": [x["ts_utc"] for x in rr],
            "ghi": np.array([x["ghi"] for x in rr], dtype=float),
            "gti": np.array([x["gti"] for x in rr], dtype=float),
            "dni": np.array([x["dni"] for x in rr], dtype=float),
            "dhi": np.array([x["dhi"] for x in rr], dtype=float),
            "temp": np.array([x["temp_air"] for x in rr], dtype=float),
        })

    return out


def build_out_model_for_day(plant: PVPlant, day, pac_real_w: np.ndarray):
    """
    Roda expected_and_mismatch para um dia (96 pontos).
    """
    details = plant.details
    mod = module_from_pvmodule(details.module)
    inv = getattr(details, "inverter", None)
    pl = plant_from_details(details, inverter=inv, use_inverter_eff=True)

    gti = day["gti"]
    ghi = day["ghi"]
    dni = day["dni"]
    dhi = day["dhi"]
    times_utc = day["times_utc"]
    tamb = day["temp"]

    gti_ok = np.isfinite(gti)
    ghi_ok = np.isfinite(ghi)
    gpoa = np.where(gti_ok, gti, np.nan)

    # se GTI faltar, tenta transposição (se seu plant tiver geo ok)
    can_transpose = (
        (pl.lat_deg is not None) and (pl.lon_deg is not None) and
        (pl.tilt_deg is not None) and (pl.azimuth_deg is not None) and
        (len(times_utc) == len(ghi)) and np.any(ghi_ok)
    )

    if can_transpose and (not np.all(gti_ok)):
        trans = transpose_ghi_to_poa_isotropic(
            ghi=ghi,
            dhi=dhi if np.any(np.isfinite(dhi)) else None,
            dni=dni if np.any(np.isfinite(dni)) else None,
            times_utc=times_utc,
            lat_deg=float(pl.lat_deg),
            lon_deg=float(pl.lon_deg),
            tilt_deg=float(pl.tilt_deg),
            azimuth_deg=float(pl.azimuth_deg),
            albedo=float(pl.albedo),
        )
        gpoa_tr = np.asarray(trans.get("g_poa"), dtype=float)
        gpoa = np.where(gti_ok, gti, gpoa_tr)
    elif not np.any(gti_ok):
        gpoa = ghi

    # assinatura pode aceitar times_utc / vdc/idc etc. — aqui focamos no treino sintético por P
    out_model = expected_and_mismatch(
        g_poa=gpoa,
        tamb_c=tamb,
        pac_real_w=pac_real_w,
        module=mod,
        plant=pl,
        g_min_valid=0.0,
        n_points=60,
        eps_w=50.0,
        dt_minutes=15.0,
        window_minutes=60.0,
        times_utc=times_utc,
    ) or {}
    return out_model


def main():
    plant_id = 2
    source_meteo = "openmeteo"  # AJUSTE
    start = date(2026, 1, 1)
    end = date(2026, 2, 1)

    plant = PVPlant.objects.get(id=plant_id)
    details = plant.details
    strings = int(getattr(details, "strings_count", 1) or 1)

    clean_days = fetch_clean_days(plant_id, start, end, source_meteo)
    if not clean_days:
        print("Sem dias limpos no intervalo.")
        return

    X_list, y_list = [], []

    # mapeamento de labels/códigos do seu gru_model
    # Ex.: DEFAULT_LABELS = ["normal","meteo_error","soiling","degradation","short_bypass","string_disconnected","partial_shading",...]
    labels = getattr(gm, "DEFAULT_LABELS", None)
    if not labels:
        raise RuntimeError("gm.DEFAULT_LABELS não encontrado (defina o vocabulário de classes).")
    label_to_idx = {name: i for i, name in enumerate(labels)}

    for day in clean_days:
        # === potência ideal sintética ===
        # Para treino, você pode usar pac_real_w como uma proxy do ideal, ou gerar a partir do próprio modelo
        # aqui: começo com uma potência “normal” sintética proporcional ao POA (apenas placeholder).
        # Melhor: gerar pac_real_w normal a partir de um pac_expected (modelo) e usar isso como base.
        g = np.nan_to_num(day["gti"], nan=0.0)
        pac_base = (g / 1000.0) * 10000.0  # placeholder (10kW). Ajuste para sua planta.

        # 1) NORMAL
        out_normal = build_out_model_for_day(plant, day, pac_base)
        X_normal = gm.build_feature_matrix_from_power_model(out_normal)
        y_normal = np.full(96, label_to_idx["normal"], dtype=int)

        X_list.append(X_normal)
        y_list.append(y_normal)

        # 2) FALHA DE STRING (reduz P proporcionalmente)
        p_string = pac_base * ((strings - 1) / strings)
        out_string = build_out_model_for_day(plant, day, p_string)
        X_string = gm.build_feature_matrix_from_power_model(out_string)
        y_string = np.full(96, label_to_idx.get("string_disconnected", label_to_idx["normal"]), dtype=int)

        X_list.append(X_string)
        y_list.append(y_string)

        # 3) SOILING (10% off no dia)
        p_soiling = pac_base * 0.90
        out_soiling = build_out_model_for_day(plant, day, p_soiling)
        X_soiling = gm.build_feature_matrix_from_power_model(out_soiling)
        y_soiling = np.full(96, label_to_idx.get("soiling", label_to_idx["normal"]), dtype=int)

        X_list.append(X_soiling)
        y_list.append(y_soiling)

        # 4) DEGRADAÇÃO (ex.: 3% fixo)
        p_deg = pac_base * 0.97
        out_deg = build_out_model_for_day(plant, day, p_deg)
        X_deg = gm.build_feature_matrix_from_power_model(out_deg)
        y_deg = np.full(96, label_to_idx.get("degradation", label_to_idx["normal"]), dtype=int)

        X_list.append(X_deg)
        y_list.append(y_deg)

    # Treino (sua função)
    art = gm.train_gru_from_series(X_series_list=X_list, y_series_list=y_list)
    print("Treino finalizado:", art)


if __name__ == "__main__":
    main()
