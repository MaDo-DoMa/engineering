package org.acme.etf.service;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.acme.etf.model.Etf;
import org.acme.etf.repository.EtfRepository;
import org.jboss.logging.Logger;

import io.quarkus.scheduler.Scheduled;

/**
 * Keeps the price history and saved predictions fresh automatically.
 *
 * Runs twice a day (UTC 05:30 - after Tiingo publishes the previous US close,
 * and UTC 22:30 - shortly after the US market closes): syncs every ETF that
 * already exists in the database and generates a new prediction for each.
 *
 * NOTE: sync is incremental and idempotent - it only fetches rows newer than
 * the latest stored date, so calling it again is cheap and safe.
 */
@ApplicationScoped
public class ScheduledSyncService {

    private static final Logger LOG = Logger.getLogger(ScheduledSyncService.class.getName());

    @Inject
    EtfRepository etfRepository;

    @Inject
    EtfSyncService syncService;

    @Inject
    EtfPredictionService predictionService;

    @Scheduled(cron = "0 30 5,22 * * ?")
    void syncAndPredictAll() {
        if (etfRepository.count() == 0) {
            LOG.info("Scheduled sync skipped: no ETFs in the database yet.");
            return;
        }
        LOG.info("Scheduled sync started.");
        for (Etf etf : etfRepository.listAll()) {
            try {
                syncService.syncEtf(etf.getSymbol());
                predictionService.generatePrediction(etf.getSymbol());
            } catch (Exception e) {
                LOG.errorf(e, "Scheduled sync failed for %s", etf.getSymbol());
            }
        }
        LOG.info("Scheduled sync finished.");
    }
}
