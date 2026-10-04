package org.acme.etf.repository;

import io.quarkus.hibernate.orm.panache.PanacheRepository;
import jakarta.enterprise.context.ApplicationScoped;
import org.acme.etf.model.Etf;

import java.util.Optional;

@ApplicationScoped
public class EtfRepository implements PanacheRepository<Etf> {
    public Optional<Etf> findBySymbolOptional(String symbol) {
        return find("symbol", symbol).firstResultOptional();
    }
}
