# Smernice za anotaciju — 8 pravnih entiteta (shema NER4Legal_SRB)

Anotira se srpska legislativa (latinica). Označava se **najduži smisleni raspon**
teksta. Ako se dvoumiš, označi uže i konzistentno. Ne preklapaj raspone.

| Oznaka | Šta obuhvata | Primeri |
|---|---|---|
| **COURT** | naziv suda | „Ustavni sud", „Vrhovni kasacioni sud", „Viši sud u Nišu" |
| **DATE** | kalendarski datum | „15. decembra 2014. godine", „1.1.2020.", „31. januara" |
| **DECISION** | konačna odluka/akt o predmetu | „presuda", „rešenje", „odluka Vlade" (u kontekstu konkretne odluke) |
| **LAW** | naziv ili skraćenica pisanog propisa | „ovog zakona", „Krivični zakonik", „Zakon o radu", „Ustav" |
| **MONEY** | novčani iznos | „100.000 dinara", „500 evra", „u iznosu od 1.000,00 RSD" |
| **OFFICIAL_GAZETTE** | objava u Sl. glasniku | „Službeni glasnik RS, broj 24/2005", „(„Sl. glasnik RS", br. 61/05)" |
| **PERSON** | ime lica (i anonimizovane skraćenice) | „Petar Petrović", „P.P." |
| **REFERENCE** | alfanumerička oznaka (predmeta/akta) | „Gž 1234/20", „broj 021-05/2019" |

## Napomene za legislativu (za razliku od presuda)
- U zakonima je **LAW najčešći** entitet („ovog zakona", „ovim zakonom", nazivi
  propisa). PERSON, DECISION i REFERENCE su retki (češći u presudama).
- Generičke institucije (ministarstvo, Narodna skupština, Vlada) **NISU** u ovoj
  shemi (osim suda → COURT). Ne označavaju se.
- Interne reference na članove („član 79.", „stav 5.") — **NE označavati** kao
  REFERENCE (to su oznake predmeta/akata, ne članova), osim ako je jasno oznaka
  akta. Ostaviti neoznačeno radi doslednosti sa NER4Legal.
- Datum unutar OFFICIAL_GAZETTE se ne izdvaja posebno — ceo navod glasnika je jedan raspon.

## Postupak
1. Otvori pred-obeleženu rečenicu u Label Studio.
2. Ispravi: dodaj promašene, obriši pogrešne, doteraj granice raspona.
3. Sačuvaj (Submit). Konzistentnost je važnija od savršenstva pojedinačnog slučaja.
