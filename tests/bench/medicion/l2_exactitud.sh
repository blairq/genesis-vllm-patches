#!/bin/bash
# El estado traido de L2 da la MISMA salida? turno 2 desde cero (x2, piso de ruido) contra rescatado de L2.
R=/home/usuario/Proyectos/genesis-vllm-patches; et=$1; shift
C="-f $R/compose/docker-compose.qwen38-27b-idiotsavant-sm86.yml -f $R/compose/ov-aislado.yml"
B=/traces/banco_pedidos_reales; U=http://127.0.0.1:8391; O=/traces/banco/exact_$et
docker stop genesis-27b-idiotsavant >/dev/null 2>&1; docker rm -f genesis-27b-pruebas >/dev/null 2>&1
(cd $R/compose && env GENESIS_KV_FLAG=--kv-transfer-config VLLM_SERVER_DEV_MODE=1 "$@" docker compose $C up -d --force-recreate >/dev/null 2>&1)
t0=$(date +%s); until curl -sf -m 3 $U/health >/dev/null 2>&1; do sleep 5; [ $(( $(date +%s) - t0 )) -gt 900 ] && break; done
echo "$(date +%T) == $et ($*)"; docker exec genesis-27b-pruebas mkdir -p $O
G() { docker exec genesis-27b-pruebas python3 /traces/banco/tokens_greedy.py 256 $B/$1 $O/$2.json; }
reset() { for i in 1 2 3 4 5 6; do r=$(curl -s -m 30 -X POST "$U/reset_prefix_cache$1"); echo "$r" | grep -q true && return; sleep 5; done; echo "reset FALLO $1"; }
espera_stores() { prev=x; for i in $(seq 1 40); do m=$(curl -s -m 5 $U/metrics | grep -E "^vllm:kv_offload_total_bytes_total" | tr -d '\n'); [ "$m" = "$prev" ] && break; prev=$m; sleep 5; done; }
G 1005_215125_0003.json cero1; reset "?reset_external=true"
G 1005_215125_0003.json cero2; reset "?reset_external=true"
G 1005_214847_0001.json t1; espera_stores; reset ""
G 1005_215125_0003.json l2
reset "?reset_external=true"; G 1005_214847_0001.json t2; G 1005_215125_0003.json local
docker exec genesis-27b-pruebas python3 -c "
import json
a = {k: json.load(open('$O/%s.json' % k)) for k in ('cero1', 'cero2', 'l2')}
def igual(x, y):
    n = 0
    while n < min(len(x), len(y)) and x[n] == y[n]: n += 1
    return n
print('cero1 vs cero2: %d de %d tokens iguales' % (igual(a['cero1'], a['cero2']), len(a['cero1'])))
print('cero1 vs l2:    %d de %d tokens iguales' % (igual(a['cero1'], a['l2']), len(a['cero1'])))
print('cero2 vs l2:    %d de %d tokens iguales' % (igual(a['cero2'], a['l2']), len(a['cero2'])))
a['local'] = json.load(open('$O/local.json'))
print('local vs l2:    %d de %d tokens iguales' % (igual(a['local'], a['l2']), len(a['l2'])))
T = {k: json.load(open('$O/%s_top.json' % k)) for k in ('cero1', 'local', 'l2')}
print('primer token (logprob del elegido): cero %s | local %s | l2 %s' % (T['cero1'][0][0], T['local'][0][0], T['l2'][0][0]))"
echo "EXACTITUD $et LISTO"
