package org.acme.etf.client;

import jakarta.ws.rs.GET;
import jakarta.ws.rs.Path;
import jakarta.ws.rs.PathParam;
import jakarta.ws.rs.QueryParam;
import org.acme.etf.client.dto.TiingoEtfPriceResponse;
import org.eclipse.microprofile.rest.client.annotation.ClientHeaderParam;
import org.eclipse.microprofile.rest.client.inject.RegisterRestClient;

import java.util.List;

@RegisterRestClient(configKey= "tiingo-api")
@ClientHeaderParam(name = "Authorization", value = "Token ${tiingo.api.key}")
public interface TiingoClient {
    @GET
    @Path("/daily/{ticker}/prices")
    List<TiingoEtfPriceResponse> getHistoricalPrices(
            @PathParam("ticker") String ticker,
            @QueryParam("startDate") String startDate
    );
}
