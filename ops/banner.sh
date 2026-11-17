#!/bin/sh
# Startup banner: waits for every service to answer (up to ~90 s each,
# polled in parallel-ish sequence) and prints the working public links.
#
# Runs as a short-lived docker compose service (see docker-compose.yml).
# Foreground display after `docker compose up -d`:  docker compose run --rm banner
# Background display:                              docker compose logs banner

wait_for() {
  label=$1
  internal_url=$2
  public_url=$3
  i=0
  while [ $i -lt 90 ]; do
    if curl -sf -o /dev/null "$internal_url"; then
      echo "  🟢 $label $public_url"
      return 0
    fi
    i=$((i + 1))
    sleep 1
  done
  echo "  🔴 $label $public_url (brak odpowiedzi po 90 s - docker compose logs)"
}

echo ""
echo "  ============================================================"
echo "     📈  SYSTEM PREDYKCJI ETF  -  usługi uruchomione"
echo "  ============================================================"
wait_for "Frontend (Streamlit):   " "http://frontend:8501"      "http://localhost:8501"
wait_for "Quarkus API (Swagger):  " "http://java-backend:8080/q/swagger-ui" "http://localhost:8080/q/swagger-ui"
wait_for "FastAPI ML (Docs):      " "http://python-ml:8000/docs" "http://localhost:8000/docs"
echo "  ⚪ Postgres:             etf_db @ localhost:5432 (user/password)"
echo "  ------------------------------------------------------------"
echo "     Logi:  docker compose logs -f <serwis>"
echo "     Stop:  docker compose down   (dane bazy ZOSTAJĄ)"
echo "  ============================================================"
echo ""
