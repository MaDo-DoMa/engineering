package org.acme.etf.repository;

import io.quarkus.hibernate.orm.panache.PanacheRepository;
import jakarta.enterprise.context.ApplicationScoped;
import org.acme.etf.model.EtfPrediction;

import java.util.List;

@ApplicationScoped
public class EtfPredictionRepository implements PanacheRepository<EtfPrediction> {
    public List<EtfPrediction> findByEtfIdOrderByTargetDateDesc(Long etfId) {
        return find("etf.id = ?1 order by targetDate desc", etfId).list();
    }
}
