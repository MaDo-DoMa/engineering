package org.acme.etf.repository;

import io.quarkus.hibernate.orm.panache.PanacheRepository;
import jakarta.enterprise.context.ApplicationScoped;
import org.acme.etf.model.EtfPriceHistory;

import java.time.LocalDate;
import java.util.List;

@ApplicationScoped
public class EtfPriceHistoryRepository implements PanacheRepository<EtfPriceHistory> {
    public List<EtfPriceHistory> findByEtfIdOrderedByDateDesc(Long etfId){
        return find("etf.id = ?1 order by date desc", etfId).list();
    }

    public LocalDate findLatestDateForEtf(Long etfId){
        return find("etf.id = ?1 order by date desc", etfId)
                .firstResultOptional()
                .map(EtfPriceHistory::getDate)
                .orElse(LocalDate.of(2000,1,1));
    }
}
