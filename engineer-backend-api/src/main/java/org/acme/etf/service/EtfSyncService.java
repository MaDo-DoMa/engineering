package org.acme.etf.service;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import jakarta.transaction.Transactional;
import org.acme.etf.client.TiingoClient;
import org.acme.etf.client.dto.TiingoEtfPriceResponse;
import org.acme.etf.model.Etf;
import org.acme.etf.model.EtfPriceHistory;
import org.acme.etf.repository.EtfPriceHistoryRepository;
import org.acme.etf.repository.EtfRepository;
import org.eclipse.microprofile.rest.client.inject.RestClient;

import java.time.LocalDate;
import java.util.List;
import org.jboss.logging.Logger;

@ApplicationScoped
public class EtfSyncService {

    private static final Logger LOG = Logger.getLogger(EtfSyncService.class);

    @Inject
    @RestClient
    TiingoClient tiingoClient;

    @Inject
    EtfRepository etfRepository;

    @Inject
    EtfPriceHistoryRepository historyRepository;

    @Transactional
    public void syncEtf(String symbol) {
        // 1. Znajdź lub utwórz ETF w bazie danych
        Etf etf = etfRepository.findBySymbolOptional(symbol).orElseGet(() -> {
            Etf newEtf = new Etf();
            newEtf.setSymbol(symbol.toUpperCase());
            etfRepository.persist(newEtf);
            return newEtf;
        });

        // 2. Ustal od jakiej daty pobrać dane (żeby uniknąć duplikatów)
        LocalDate lastDate = historyRepository.findLatestDateForEtf(etf.getId());
        LocalDate startDate = lastDate.plusDays(1);

        if (startDate.isAfter(LocalDate.now())) {
            LOG.infof("Data for %s is already fully up-to-date.", symbol);
            return;
        }

        // 3. Pobranie danych z zewnętrznego API
        LOG.infof("Fetching history for %s from Tiingo starting from date: %s", symbol, startDate);
        List<TiingoEtfPriceResponse> prices = tiingoClient.getHistoricalPrices(symbol, startDate.toString());

        // 4. Zapis nowych notowań do bazy
        for (TiingoEtfPriceResponse dto : prices) {
            EtfPriceHistory history = new EtfPriceHistory();
            history.setEtf(etf);
            history.setDate(dto.date);
            history.setOpenPrice(dto.open);
            history.setHighPrice(dto.high);
            history.setLowPrice(dto.low);
            history.setClosePrice(dto.close);
            history.setVolume(dto.volume);

            historyRepository.persist(history);
        }

        LOG.infof("Success! Saved %d new records for %s.", prices.size(), symbol);
    }
}
