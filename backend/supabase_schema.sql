-- Run this in Supabase SQL editor to create the table
CREATE TABLE IF NOT EXISTS drug_reviews (
  id SERIAL PRIMARY KEY,
  "uniqueID" INTEGER,
  "drugName" TEXT NOT NULL,
  "condition" TEXT,
  review TEXT,
  rating FLOAT,
  date TEXT,
  "usefulCount" INTEGER DEFAULT 0,
  response_category INTEGER
);

-- Index for fast drug+condition lookups
CREATE INDEX IF NOT EXISTS idx_drug_condition
ON drug_reviews ("drugName", "condition");

-- Index for drug name search
CREATE INDEX IF NOT EXISTS idx_drug_name
ON drug_reviews ("drugName");
