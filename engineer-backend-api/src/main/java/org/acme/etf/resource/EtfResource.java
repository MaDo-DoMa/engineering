package org.acme.etf.resource;


import jakarta.inject.Inject;
import jakarta.ws.rs.*;
import jakarta.ws.rs.core.MediaType;
import jakarta.ws.rs.core.Response;
import org.acme.etf.model.Etf;
import org.acme.etf.repository.EtfPredictionRepository;
import org.acme.etf.repository.EtfPriceHistoryRepository;
import org.acme.etf.repository.EtfRepository;
import org.acme.etf.service.EtfPredictionService;
import org.acme.etf.service.EtfSyncService;
import org.eclipse.microprofile.openapi.annotations.Operation;
import org.eclipse.microprofile.openapi.annotations.tags.Tag;

import java.util.List;
import java.util.Locale;

@Path("/api/etfs")
@Produces(MediaType.APPLICATION_JSON)
@Consumes(MediaType.APPLICATION_JSON)
@Tag(name = "ETF API", description ="ETF management, price history, and predictions")
public class EtfResource {
    @Inject
    EtfRepository etfRepository;

    @Inject
    EtfPriceHistoryRepository  historyRepository;

    @Inject
    EtfPredictionRepository etfPredictionRepository;

    @Inject
    EtfSyncService syncService;

    @Inject
    EtfPredictionService predictionService;

    @GET
    @Operation(summary ="Get a list of all ETFs present in the database")
    public List<Etf> getAllEtfs(){
        return etfRepository.listAll();
    }

    @GET
    @Path("/{symbol}/history")
    @Operation(summary="Get price history for a given symbol")
    public Response getHistory(@PathParam("symbol")  String symbol){
        return etfRepository.findBySymbolOptional(symbol.toUpperCase())
                .map(etf-> Response.ok(historyRepository.findByEtfIdOrderedByDateDesc(etf.getId())).build())
                .orElse(Response.status(Response.Status.NOT_FOUND).build());
    }

    @GET
    @Path("/{symbol}/predictions")
    @Operation(summary = "Get generated predictions for a given symbol")
    public Response getPredictions(@PathParam("symbol")  String symbol){
        return etfRepository.findBySymbolOptional(symbol.toUpperCase())
                .map(etf -> Response.ok(etfPredictionRepository.findByEtfIdOrderByTargetDateDesc(etf.getId())).build())
                .orElse(Response.status(Response.Status.NOT_FOUND).build());
    }

    @POST
    @Path("/{symbol}/sync")
    @Operation(summary ="Force fetch latest data from Tiingo and generate a new prediction")
    public Response syncAndPredict(@PathParam("symbol") String symbol){
        String upperSymbol = symbol.toUpperCase();

        syncService.syncEtf(upperSymbol);
        predictionService.generatePrediction(upperSymbol);

        return Response.ok().entity("{\"status\": \"Successfully synchronized and generated predictions for " + upperSymbol + "\"}").build();
    }

}
