package org.acme.etf.service;

import java.util.Map;

/**
 * Static mapping between frontend-facing symbols and real Tiingo tickers.
 *
 * The database/REST layer ALWAYS uses the frontend symbol as the canonical
 * identifier (SPY500, NASDAQ, ...), so the frontend can keep sending
 * human-readable names. Only the outgoing Tiingo HTTP request is mapped.
 */
public final class TickerMapping {

    private TickerMapping() {
        // static utility class - no instances
    }

    /**
     * frontendSymbol -> Tiingo ticker.
     * WIG20 -> EPOL: Tiingo has no GPW/WIG20 coverage; EPOL (iShares MSCI
     * Poland ETF) tracks the Polish market and is available in Tiingo.
     * INVESCO -> RSP: "Invesco" is an issuer, not an index; RSP is the
     * Invesco S&P 500 Equal Weight fund.
     */
    private static final Map<String, String> FRONTEND_TO_TIINGO = Map.of(
            "SPY500", "SPY",
            "NASDAQ", "QQQ",
            "GLD", "GLD",
            "WIG20", "EPOL",
            "INVESCO", "RSP"
    );

    /**
     * frontendSymbol -> full, human-readable fund name persisted in the
     * "etfs.name" column (used by GET /api/etfs and any UI listing).
     */
    private static final Map<String, String> DISPLAY_NAMES = Map.of(
            "SPY500", "SPDR S&P 500 ETF Trust",
            "NASDAQ", "Invesco QQQ Trust (Nasdaq-100)",
            "GLD", "SPDR Gold Shares",
            "WIG20", "iShares MSCI Poland ETF (WIG20 proxy)",
            "INVESCO", "Invesco S&P 500 Equal Weight"
    );

    public static String toTiingo(String frontendSymbol) {
        String upper = frontendSymbol.toUpperCase();
        return FRONTEND_TO_TIINGO.getOrDefault(upper, upper);
    }

    public static String displayName(String frontendSymbol) {
        String upper = frontendSymbol.toUpperCase();
        return DISPLAY_NAMES.getOrDefault(upper, upper);
    }
}
