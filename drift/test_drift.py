"""Prueba automatizada de data drift (pytest): verde con datos del mismo origen, rojo con datos derivados.
USO: bash scripts/entrenar.sh -m pytest -v drift/test_drift.py
"""
import datos
from data_drift import estado_global, evaluar, lote_derivado

BASE, MISMO_ORIGEN = datos.baseline_y_mismo_origen()


def test_mismo_origen_pasa_en_verde():
    tabla = evaluar(BASE, MISMO_ORIGEN)
    assert estado_global(tabla) == "VERDE", tabla[tabla.estado != "VERDE"]


def test_lote_derivado_falla_en_rojo():
    tabla = evaluar(BASE, lote_derivado(MISMO_ORIGEN))
    rojas = set(tabla[tabla.estado == "ROJO"].variable)
    assert estado_global(tabla) == "ROJO"
    assert {"monto_total", "pct_inicial", "plazo_cat", "con_plan"} <= rojas, rojas


def test_variables_no_tocadas_siguen_estables():
    tabla = evaluar(BASE, lote_derivado(MISMO_ORIGEN)).set_index("variable")
    assert tabla.loc["n_creditos_previos", "estado"] == "VERDE" and tabla.loc["atraso_previo", "estado"] == "VERDE"
