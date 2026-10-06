package org.acme.etf.service;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import jakarta.transaction.Transactional;
import org.acme.etf.client.ml.MlPredictionRequest;
import org.acme.etf.client.ml.MlPredictionResponse;
import org.acme.etf.client.ml.MlPriceData;
import org.acme.etf.client.ml.MlServiceClient;
import org.acme.etf.model.Etf;
import org.acme.etf.model.EtfPrediction;
import org.acme.etf.model.EtfPriceHistory;
import org.acme.etf.repository.EtfPredictionRepository;
import org.acme.etf.repository.EtfPriceHistoryRepository;
import org.acme.etf.repository.EtfRepository;
import org.eclipse.microprofile.rest.client.inject.RestClient;

import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

import org.jboss.logging.Logger;

@ApplicationScoped
public class EtfPredictionService {
    private static final Logger LOG = Logger.getLogger(EtfPredictionService.class.getName());

    @Inject
    EtfRepository etfRepository;

    @Inject
    EtfPriceHistoryRepository historyRepository;

    @Inject
    EtfPredictionRepository predictionRepository;

    @Inject
    @RestClient
    MlServiceClient mlServiceClient;

    @Transactional
    public void generatePrediction(String symbol){
        Optional<Etf> etfOpt = etfRepository.findBySymbolOptional(symbol);
        if (etfOpt.isEmpty()){
            LOG.errorf("ETF %s not found in database.", symbol);
            return;
        }
        Etf etf = etfOpt.get();

        List<EtfPriceHistory> history = historyRepository.findByEtfIdOrderedByDateDesc(etf.getId());
        if(history.size() <10){
            LOG.warnf("Not enough historical data to generate prediction for %s", symbol);
            return;
        }

        List<MlPriceData> priceDataList = history.stream()
                .sorted((h1, h2) -> h1.getDate().compareTo(h2.getDate()))
                .map(h -> new MlPriceData(h.getDate(), h.getClosePrice(), h.getVolume()))
                .collect(Collectors.toList());

        MlPredictionRequest request = new MlPredictionRequest(symbol,  priceDataList);

        try{
            LOG.infof("Requesting prediction from ML service for %s...", symbol);
            MlPredictionResponse response = mlServiceClient.getPrediction(request);

            EtfPrediction prediction = new EtfPrediction();
            prediction.setEtf(etf);
            prediction.setTargetDate(response.targetDate);
            prediction.setPredictedClosePrice(response.predictedClosePrice);

            predictionRepository.persist(prediction);
            LOG.infof("Prediction for %s successfully generated and saved: %s", symbol,  response.predictedClosePrice);
        }catch (Exception e){
            LOG.errorf(e, "Error communicating with ML service for %s", symbol);
        }
    }
}
