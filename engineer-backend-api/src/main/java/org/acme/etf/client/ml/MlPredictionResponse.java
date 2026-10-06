package org.acme.etf.client.ml;

import java.math.BigDecimal;
import java.time.LocalDate;

public class MlPredictionResponse {
    public LocalDate targetDate;
    public BigDecimal predictedClosePrice;
}
