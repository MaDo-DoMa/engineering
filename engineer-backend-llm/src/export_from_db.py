import pandas as pd
from sqlalchemy import create_engine
from config import DATA_DIR

# Jawnie wskazujemy sterownik psycopg2 w adresie URL
DB_URL = "postgresql+psycopg2://quarkus:quarkus@localhost:63932/quarkus"


def export_spy_data():
    print("Connecting to PostgreSQL database...")
    engine = create_engine(DB_URL)

    # Zapytanie pobierające historię dla SPY
    query = """
            SELECT eph.date, \
                   eph.openprice  AS "openPrice", \
                   eph.highprice  AS "highPrice", \
                   eph.lowprice   AS "lowPrice", \
                   eph.closeprice AS "closePrice", \
                   eph.volume
            FROM etf_price_history eph
                     JOIN etfs e ON eph.etf_if = e.id
            WHERE e.symbol = 'SPY'
            ORDER BY eph.date ASC; \
            """

    print("Fetching historical data for SPY...")
    df = pd.read_sql(query, engine)

    if df.empty:
        print("Warning: No records found for SPY in the database!")
        return

    # Zapis do pliku CSV w folderze data/
    output_path = DATA_DIR / "sample_etf.csv"
    df.to_csv(output_path, index=False)
    print(f"Success! Exported {len(df)} rows to {output_path}")


if __name__ == "__main__":
    export_spy_data()