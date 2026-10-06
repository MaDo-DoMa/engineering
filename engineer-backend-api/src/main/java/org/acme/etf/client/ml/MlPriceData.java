package org.acme.etf.client.ml;

import java.math.BigDecimal;
import java.time.LocalDate;

public class MlPriceData {
    public LocalDate date;
    public BigDecimal closePrice;
    public Long volume;

    public MlPriceData(LocalDate date, BigDecimal closePrice, Long volume) {
        this.date = date;
        this.closePrice = closePrice;
        this.volume = volume;
    }
}
