package org.acme.etf.model;

import jakarta.persistence.*;

import java.math.BigDecimal;
import java.time.LocalDate;

public class EtfPrediction {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "etf_id", nullable = false)
    private Etf etf;

    @Column(nullable = false)
    private LocalDate targetDate;

    @Column(precision = 19, scale = 4, nullable = false)
    private BigDecimal predictedClosePrice;

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public Etf getEtf() {
        return etf;
    }

    public void setEtf(Etf etf) {
        this.etf = etf;
    }

    public LocalDate getTargetDate() {
        return targetDate;
    }

    public void setTargetDate(LocalDate targetDate) {
        this.targetDate = targetDate;
    }

    public BigDecimal getPredictedClosePrice() {
        return predictedClosePrice;
    }

    public void setPredictedClosePrice(BigDecimal predictedClosePrice) {
        this.predictedClosePrice = predictedClosePrice;
    }
}
