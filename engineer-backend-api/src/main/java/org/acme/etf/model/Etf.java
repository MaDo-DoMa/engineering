package org.acme.etf.model;

import jakarta.persistence.*;

@Entity
@Table(name="etfs")
public class Etf {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(unique = true, nullable = false)
    private String symbol;

    private String name;

    public Long getId() {return id;}
    public void setId(Long id) {this.id = id;}

    public String getSymbol() {return symbol;}
    public void setSymbol(String symbol) {this.symbol = symbol;}

    public String getName() {return name;}
    public void setName(String name) {this.name = name;}
}
