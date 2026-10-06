package org.acme.etf.client.ml;

import java.util.List;

public class MlPredictionRequest {
    public String symbol;
    public List<MlPriceData> history;

    public MlPredictionRequest(String symbol, List<MlPriceData> history) {
        this.symbol = symbol;
        this.history = history;
    }
}
