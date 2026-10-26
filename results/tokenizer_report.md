# ScamLens Phase 3 - tokenizer comparison

183 random TRAINING messages, sampled per `script_style` ({'roman_hinglish': 50, 'devanagari': 50, 'mixed': 33, 'english': 50}). Lower tokens/word and a lower single-character share is usually better for Hinglish.

| model | avg tokens/message | avg tokens/word | single-char share | truncated at 128 | status |
|---|---|---|---|---|---|
| `google/muril-base-cased` | 33.4 | 1.55 | 21.9% | 0.0% | loaded |
| `xlm-roberta-base` | 34.6 | 1.61 | 25.7% | 0.0% | loaded |
| `bert-base-multilingual-cased` | 43.7 | 2.03 | 23.9% | 0.0% | loaded |
| `l3cube-pune/hing-bert` | 46.9 | 2.17 | 28.4% | 0.0% | loaded |

## By script style (up to 100 training rows per style)

| model | style | rows | tokens/message | tokens/word | single-char share |
|---|---|---|---|---|---|
| `google/muril-base-cased` | roman_hinglish | 100 | 46.1 | 1.61 | 20.4% |
| `google/muril-base-cased` | devanagari | 100 | 28.1 | 1.38 | 20.6% |
| `google/muril-base-cased` | mixed | 33 | 29.7 | 1.48 | 21.6% |
| `google/muril-base-cased` | english | 100 | 39.1 | 1.66 | 27.4% |
| `xlm-roberta-base` | roman_hinglish | 100 | 48.6 | 1.70 | 22.2% |
| `xlm-roberta-base` | devanagari | 100 | 30.6 | 1.51 | 25.8% |
| `xlm-roberta-base` | mixed | 33 | 31.8 | 1.58 | 26.2% |
| `xlm-roberta-base` | english | 100 | 39.3 | 1.67 | 30.3% |
| `bert-base-multilingual-cased` | roman_hinglish | 100 | 55.2 | 1.92 | 17.9% |
| `bert-base-multilingual-cased` | devanagari | 100 | 47.9 | 2.36 | 27.7% |
| `bert-base-multilingual-cased` | mixed | 33 | 47.4 | 2.36 | 27.8% |
| `bert-base-multilingual-cased` | english | 100 | 41.7 | 1.77 | 25.8% |
| `l3cube-pune/hing-bert` | roman_hinglish | 100 | 55.1 | 1.92 | 15.1% |
| `l3cube-pune/hing-bert` | devanagari | 100 | 63.1 | 3.11 | 36.4% |
| `l3cube-pune/hing-bert` | mixed | 33 | 60.5 | 3.01 | 34.8% |
| `l3cube-pune/hing-bert` | english | 100 | 36.9 | 1.56 | 27.8% |

## Same 5 Hinglish messages, different tokenizers

These are actual `roman_hinglish` TRAINING rows (synthetic/scrubbed).

**Message 1** (26 words, 134 chars)

* **`google/muril-base-cased`** (49 tokens): W | ##aq | ##t | se | pehle | or | na | ##see | ##b | se | zyada | kisi | ko | kuch | nahi | mil | ##ta | , | Z | ##ind | ##gi | wo | nahi | he | jo | hum | soch | ##te | hai | Z | ##ind | ##gi | wo | hai | jo | ham | jeet | ##ey | hai | . | . | . | . | . | . | . | . | . | .
* **`xlm-roberta-base`** (41 tokens): ▁Waqt | ▁se | ▁pe | hle | ▁or | ▁na | se | eb | ▁se | ▁z | yada | ▁kisi | ▁ko | ▁kuch | ▁nahi | ▁mil | ta | , | Z | ind | gi | ▁wo | ▁nahi | ▁he | ▁jo | ▁hum | ▁so | chte | ▁hai | ▁Zin | d | gi | ▁wo | ▁hai | ▁jo | ▁ham | ▁je | ete | y | ▁hai | ..........
* **`bert-base-multilingual-cased`** (55 tokens): W | ##aq | ##t | se | pe | ##hle | or | nas | ##ee | ##b | se | z | ##yada | kis | ##i | ko | ku | ##ch | nahi | mil | ##ta | , | Zi | ##nd | ##gi | wo | nahi | he | jo | hu | ##m | so | ##chte | hai | Zi | ##nd | ##gi | wo | hai | jo | ham | je | ##ete | ##y | hai | . | . | . | . | . | . | . | . | . | .
* **`l3cube-pune/hing-bert`** (59 tokens): wa | ##q | ##t | se | pe | ##hl | ##e | or | nas | ##ee | ##b | se | z | ##yad | ##a | ki | ##si | ko | ku | ##ch | nah | ##i | mil | ##ta | , | z | ##ind | ##gi | wo | nah | ##i | he | jo | hum | soc | ##ht | ##e | hai | z | ##ind | ##gi | wo | hai | jo | ham | je | ##ete | ##y | hai | . | . | . | . | . | . | . | . | . | .

**Message 2** (39 words, 197 chars)

* **`google/muril-base-cased`** (58 tokens): Priya | G | ##ra | hak | , | A | ##ap | ##ke | case | k | ##3 | baare | mein | ek | update | hai | . | A | ##ap | ka | courier | par | ##c | el | del | ##iv | er | nahi | ho | pay | ##a | . | Par | ##cel | paan | ke | ly | ##ie | < | URL | > | par | address | update | kare | ##i | . | n | Yeh | message | kisi | ke | saath | share | na | kare | in | .
* **`xlm-roberta-base`** (59 tokens): ▁Pri | ya | ▁Gra | ▁hak | , | ▁Aap | ke | ▁case | ▁k | 3 | ▁baar | e | ▁mein | ▁ek | ▁update | ▁hai | . | ▁Aap | ▁ka | ▁cour | ier | ▁parc | ▁el | ▁de | liv | ▁er | ▁nahi | ▁ho | ▁pa | ya | . | ▁Parce | l | ▁pa | an | ▁ke | ▁ly | ie | ▁< | URL | > | ▁par | ▁address | ▁update | ▁kare | i | . | n | ▁Yeh | ▁message | ▁kisi | ▁ke | ▁saat | h | ▁share | ▁na | ▁kare | ▁in | .
* **`bert-base-multilingual-cased`** (65 tokens): Pri | ##ya | G | ##ra | hak | , | Aa | ##p | ##ke | case | k | ##3 | ba | ##are | mein | ek | update | hai | . | Aa | ##p | ka | cour | ##ier | parc | el | del | ##iv | er | nahi | ho | pay | ##a | . | Parc | ##el | pa | ##an | ke | ly | ##ie | < | URL | > | par | address | update | kar | ##ei | . | n | Ye | ##h | message | kis | ##i | ke | saat | ##h | share | na | kar | ##e | in | .
* **`l3cube-pune/hing-bert`** (68 tokens): pri | ##ya | gr | ##a | ha | ##k | , | aa | ##p | ##ke | case | k | ##3 | ba | ##are | mein | ek | update | hai | . | aa | ##p | ka | courier | parc | el | del | ##iv | er | nah | ##i | ho | pay | ##a | . | parcel | pa | ##an | ke | l | ##yi | ##e | < | ur | ##l | > | par | address | update | ka | ##re | ##i | . | n | ye | ##h | message | ki | ##si | ke | sa | ##ath | share | na | ka | ##re | in | .

**Message 3** (42 words, 240 chars)

* **`google/muril-base-cased`** (61 tokens): Nam | ##aste | ji | , | Is | service | message | par | d | ##hyan | dei | ##n | . | Che | ##ta | ##wani | : | aapke | case | ke | liye | recovery | agent | assign | hua | . | Loan | file | band | karne | ke | liye | Rs | 1 | , | 200 | closure | fee | bheje | ##in | . | Yeh | message | kisi | ke | saath | share | na | kare | ##in | . | Is | message | ko | sam | ##bha | ##l | kar | rakhe | ##in | .
* **`xlm-roberta-base`** (64 tokens): ▁Nama | ste | ▁ji | , | ▁Is | ▁service | ▁message | ▁par | ▁dh | yan | ▁de | in | . | ▁Che | tawan | i | : | ▁aap | ke | ▁case | ▁ke | ▁liye | ▁recovery | ▁agent | ▁assign | ▁hua | . | ▁Loan | ▁file | ▁band | ▁karne | ▁ke | ▁liye | ▁Rs | ▁1 | , | 200 | ▁ | closure | ▁fee | ▁bhe | je | in | . | ▁Yeh | ▁message | ▁kisi | ▁ke | ▁saat | h | ▁share | ▁na | ▁kare | in | . | ▁Is | ▁message | ▁ko | ▁sam | bhal | ▁kar | ▁rakh | ein | .
* **`bert-base-multilingual-cased`** (74 tokens): Nama | ##ste | ji | , | Is | service | message | par | d | ##hya | ##n | dei | ##n | . | Che | ##taw | ##ani | : | aa | ##p | ##ke | case | ke | li | ##ye | recovery | agent | ass | ##ign | hu | ##a | . | Loan | file | band | karne | ke | li | ##ye | R | ##s | 1 | , | 200 | closure | fee | b | ##hej | ##ein | . | Ye | ##h | message | kis | ##i | ke | saat | ##h | share | na | kar | ##ein | . | Is | message | ko | sam | ##bh | ##al | kar | rak | ##hei | ##n | .
* **`l3cube-pune/hing-bert`** (75 tokens): nam | ##ast | ##e | ji | , | is | service | message | par | dh | ##yan | dei | ##n | . | chet | ##awan | ##i | : | aa | ##p | ##ke | case | ke | li | ##ye | recovery | agent | assign | hua | . | loan | file | band | ka | ##rne | ke | li | ##ye | rs | 1 | , | 200 | closure | fee | b | ##he | ##je | ##in | . | ye | ##h | message | ki | ##si | ke | sa | ##ath | share | na | ka | ##re | ##in | . | is | message | ko | sam | ##bha | ##l | ka | ##r | ra | ##kh | ##ein | .

**Message 4** (38 words, 210 chars)

* **`google/muril-base-cased`** (63 tokens): Priya | G | ##raha | ##k | , | Din | khat | ##am | hone | se | pehle | jawab | dei | ##n | . | Bhai | ne | gal | ##ti | se | Rs | 9 | , | 999 | bheja | hai | , | wa | ##pas | kar | do | . | Ji | ##s | number | se | aay | ##a | hai | us | ##i | number | par | wa | ##pas | bhej | ##o | . | Sir | ##f | official | process | follow | kare | ##in | . | K | ##rip | ##ya | jaldi | kare | ##in | .
* **`xlm-roberta-base`** (65 tokens): ▁Pri | ya | ▁Gra | hak | , | ▁Din | ▁kha | tam | ▁hon | e | ▁se | ▁pe | hle | ▁jawab | ▁de | in | . | ▁Bhai | ▁ne | ▁gal | ti | ▁se | ▁Rs | ▁9 | , | 999 | ▁bhe | ja | ▁hai | , | ▁wa | pas | ▁kar | ▁do | . | ▁Jis | ▁number | ▁se | ▁a | aya | ▁hai | ▁usi | ▁number | ▁par | ▁wa | pas | ▁bhe | jo | . | ▁Sir | f | ▁official | ▁process | ▁follow | ▁kare | in | . | ▁Kri | py | a | ▁ja | ldi | ▁kare | in | .
* **`bert-base-multilingual-cased`** (72 tokens): Pri | ##ya | G | ##rah | ##ak | , | Din | k | ##hat | ##am | hon | ##e | se | pe | ##hle | jawab | dei | ##n | . | B | ##hai | ne | galt | ##i | se | R | ##s | 9 | , | 999 | b | ##hej | ##a | hai | , | wa | ##pas | kar | do | . | Jis | number | se | aa | ##ya | hai | us | ##i | number | par | wa | ##pas | b | ##hej | ##o | . | Sir | ##f | official | process | follow | kar | ##ein | . | Kr | ##ip | ##ya | ja | ##ldi | kar | ##ein | .
* **`l3cube-pune/hing-bert`** (78 tokens): pri | ##ya | gr | ##aha | ##k | , | din | k | ##hat | ##am | hon | ##e | se | pe | ##hl | ##e | jaw | ##ab | dei | ##n | . | b | ##hai | ne | gal | ##ti | se | rs | 9 | , | 999 | b | ##he | ##ja | hai | , | wa | ##pas | ka | ##r | do | . | ji | ##s | number | se | aa | ##ya | hai | us | ##i | number | par | wa | ##pas | b | ##he | ##jo | . | sir | ##f | official | process | follow | ka | ##re | ##in | . | k | ##rip | ##ya | ja | ##ld | ##i | ka | ##re | ##in | .

**Message 5** (15 words, 88 chars)

* **`google/muril-base-cased`** (46 tokens): K | ##RI | ##P | ##YA | IS | ME | ##SS | ##AG | ##E | KO | D | ##H | ##YA | ##N | SE | PA | ##DH | ##E | ##IN | . | AA | ##K | ##HR | ##I | NOT | ##ICE | : | AA | ##J | RE | - | K | ##YC | KA | ##RE | ##IN | WA | ##RNA | A | / | C | B | ##L | ##OC | ##K | .
* **`xlm-roberta-base`** (38 tokens): ▁K | RIP | YA | ▁IS | ▁M | ESS | AGE | ▁KO | ▁DH | YAN | ▁SE | ▁P | ADH | EIN | . | ▁AA | KH | RI | ▁NOT | ICE | : | ▁A | AJ | ▁RE | - | KY | C | ▁KAR | EIN | ▁W | ARNA | ▁A | / | C | ▁B | LOC | K | .
* **`bert-base-multilingual-cased`** (47 tokens): K | ##RI | ##P | ##Y | ##A | IS | ME | ##SS | ##AG | ##E | KO | DH | ##Y | ##AN | SE | PA | ##DH | ##E | ##IN | . | AA | ##K | ##H | ##RI | NO | ##TI | ##CE | : | AA | ##J | RE | - | K | ##Y | ##C | KA | ##RE | ##IN | W | ##AR | ##NA | A | / | C | BL | ##OCK | .
* **`l3cube-pune/hing-bert`** (33 tokens): k | ##rip | ##ya | is | message | ko | dh | ##yan | se | pad | ##hein | . | aa | ##kh | ##ri | notice | : | aa | ##j | re | - | ky | ##c | ka | ##re | ##in | warn | ##a | a | / | c | block | .

## Plain-English read

Lowest token inflation: **`google/muril-base-cased`** (1.55 tokens per word). A lower number means fewer pieces per Hinglish word, so more of the 128-token budget is spent on meaning. Tokenizer efficiency is only one factor: model pretraining data matters at least as much, which is why E1/E3 compare the models themselves.
