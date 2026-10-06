package org.acme.etf.client.ml;

import jakarta.ws.rs.POST;
import jakarta.ws.rs.Path;
import org.eclipse.microprofile.rest.client.inject.RegisterRestClient;

@RegisterRestClient(configKey = "ml-api")
public interface MlServiceClient {
    @POST
    @Path("/predict")
    MlPredictionResponse getPrediction(MlPredictionRequest request);
}
