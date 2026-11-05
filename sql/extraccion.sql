-- Extracción de créditos para el proyecto de MLOps. Solo lectura, sin datos personales.
SET statement_timeout='120s';
COPY (
WITH v AS (
  SELECT idventa, idcliente, fecha::date f, totalpagar::numeric t, diaspagar n, descuento
  FROM venta WHERE estado='V' AND idtipodocumento<>2 AND fecha::date<=:'corte' AND coalesce(totalpagar,0)>0
), c AS (
  SELECT v.*, CASE WHEN n<=24 THEN (f+make_interval(months=>n))::date ELSE f+n END venc FROM v WHERE n>0
), b AS (
  SELECT c.*,
    (SELECT coalesce(sum(p.importe),0) FROM pago p WHERE p.idventa=c.idventa AND p.estado='1' AND p.fecha::date BETWEEN '2015-01-01' AND LEAST((c.venc+30), :'corte'::date)) a30,
    (SELECT min(s.fp) FROM (SELECT p.fecha::date fp, sum(p.importe) OVER (ORDER BY p.fecha::date) ac FROM pago p WHERE p.idventa=c.idventa AND p.estado='1' AND p.fecha::date BETWEEN '2015-01-01' AND :'corte') s WHERE s.ac+0.005>=c.t) saldado,
    (SELECT coalesce(sum(p.importe),0) FROM pago p WHERE p.idventa=c.idventa AND p.estado='1' AND p.fecha::date=c.f) inicial
  FROM c
), e AS (SELECT b.*, (venc+30<=:'corte'::date) madura, (a30+0.005<t) atrasada FROM b)
SELECT md5(e.idventa::text||:'sal') credito_id, md5(e.idcliente::text||:'sal') cliente_id,
  e.f fecha_venta, extract(year FROM e.f)::int anio, CASE WHEN e.idventa>=4069 THEN 'ionic' ELSE 'java' END era,
  round(e.t,2) monto_total, e.n plazo_raw,
  CASE WHEN e.n>24 THEN 'dias' WHEN e.n=1 THEN '1' WHEN e.n<=3 THEN '2-3' WHEN e.n<=6 THEN '4-6' ELSE '7+' END plazo_cat,
  round(least(greatest(e.inicial/e.t,0),1),4) pct_inicial,
  CASE WHEN EXISTS (SELECT 1 FROM planpago pp WHERE pp.idventa=e.idventa) THEN 'si' ELSE 'no' END con_plan,
  (SELECT count(*) FROM detalleventa d WHERE d.idventa=e.idventa) n_lineas,
  coalesce(pr.n_prev,0) n_creditos_previos, round(coalesce(pr.monto,0),2) monto_previo,
  coalesce(pr.n_mad,0) n_previos_maduros, coalesce(pr.n_mora,0) n_previos_mora30,
  coalesce(ul.a,'sin_historial') atraso_previo,
  coalesce(pr.max_atraso,0) max_dias_atraso_previo, round(coalesce(pr.abierto,0),2) saldo_abierto_previo,
  coalesce((e.f-(SELECT min(x.f) FROM v x WHERE x.idcliente=e.idcliente AND x.f<e.f))/30,0) meses_desde_primera_venta,
  (SELECT count(*) FROM v x WHERE x.idcliente=e.idcliente AND x.f<e.f AND x.n=0) n_contado_previos,
  EXISTS (SELECT 1 FROM regularizacion_saldos_20261014 r WHERE r.tabla='venta' AND r.idfila=e.idventa) regularizada,
  e.madura, CASE WHEN e.madura THEN e.atrasada::int END mora30
FROM e
LEFT JOIN LATERAL (
  SELECT count(*) n_prev, sum(o.t) monto,
    count(*) FILTER (WHERE o.venc+30<e.f) n_mad, count(*) FILTER (WHERE o.venc+30<e.f AND o.atrasada) n_mora,
    max(greatest(CASE WHEN o.saldado<e.f THEN o.saldado-o.venc WHEN o.venc<e.f THEN e.f-o.venc ELSE 0 END,0)) max_atraso,
    sum(greatest(o.t-(SELECT coalesce(sum(p.importe),0) FROM pago p WHERE p.idventa=o.idventa AND p.estado='1' AND p.fecha::date BETWEEN '2015-01-01' AND e.f-1),0)) abierto
  FROM e o WHERE o.idcliente=e.idcliente AND o.f<e.f) pr ON true
LEFT JOIN LATERAL (
  SELECT CASE WHEN o.atrasada THEN 'si' ELSE 'no' END a FROM e o WHERE o.idcliente=e.idcliente AND o.venc+30<e.f ORDER BY o.venc DESC, o.idventa DESC LIMIT 1) ul ON true
ORDER BY e.f, e.idventa
) TO STDOUT WITH CSV HEADER;
