package org.acme.etf.model;

import jakarta.persistence.*;

import java.math.BigDecimal;
import java.time.LocalDate;
@Entity
@Table(name = "etf_price_history", uniqueConstraints = {
        @UniqueConstraint(columnNames = {"etf_id", "date"})
})
public class EtfPriceHistory {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "etf_if", nullable = false)
    private Etf etf;

    @Column(nullable = false)
    private LocalDate date;

    @Column(precision=19, scale = 4)
    private BigDecimal openPrice;

    @Column(precision=19, scale = 4)
    private BigDecimal highPrice;

    @Column(precision=19, scale = 4)
    private BigDecimal lowPrice;

    @Column(precision=19, scale = 4)
    private BigDecimal closePrice;

    private Long volume;

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

    public LocalDate getDate() {
        return date;
    }

    public void setDate(LocalDate date) {
        this.date = date;
    }

    public BigDecimal getOpenPrice() {
        return openPrice;
    }

    public void setOpenPrice(BigDecimal openPrice) {
        this.openPrice = openPrice;
    }

    public BigDecimal getHighPrice() {
        return highPrice;
    }

    public void setHighPrice(BigDecimal highPrice) {
        this.highPrice = highPrice;
    }

    public BigDecimal getLowPrice() {
        return lowPrice;
    }

    public void setLowPrice(BigDecimal lowPrice) {
        this.lowPrice = lowPrice;
    }

    public BigDecimal getClosePrice() {
        return closePrice;
    }

    public void setClosePrice(BigDecimal closePrice) {
        this.closePrice = closePrice;
    }

    public Long getVolume() {
        return volume;
    }

    public void setVolume(Long volume) {
        this.volume = volume;
    }
}
