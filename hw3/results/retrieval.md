# Retrieval evaluation

- prompts: `prompts/eval_prompts.jsonl`; date 2026-11-26
- hit@k: an expected document is among the first k passages (k = 5)
- lexical settings: `{"scheme": "bm25", "k1": 1.2, "b": 0.75, "passage_words": 150}`
- not built: vector, hybrid

## Summary

| retriever | group | n | hit@1 | hit@5 | mean ms |
|---|---|---|---|---|---|
| lexical | all | 18 | 14/18 (78%) | 16/18 (89%) | 40.6 |
| lexical | type=collection | 12 | 8/12 (67%) | 10/12 (83%) | 36.7 |
| lexical | type=special | 6 | 6/6 (100%) | 6/6 (100%) | 48.4 |
| lexical | lang=en | 16 | 14/16 (88%) | 16/16 (100%) | 44.2 |
| lexical | lang=zh | 2 | 0/2 (0%) | 0/2 (0%) | 11.6 |

## Per prompt: lexical

| id | type | lang | prompt | expected | rank | top 5 |
|---|---|---|---|---|---|---|
| P01 | special | en | 115598095 | 13639, 23925 | 1 | 13639, 23925 |
| P02 | special | en | What is the project codename for 115598095? | 13639, 23925 | 1 | 13639, 23925, 13964, 3957, 3068 |
| P03 | special | en | 115598063 | 10810, 13964 | 1 | 13964, 10810 |
| P04 | special | en | What is the project codename for 115598063? | 10810, 13964 | 1 | 13964, 10810, 3957, 13639, 3068 |
| P05 | special | en | zorvian lantern | 3350, 3800 | 1 | 3350, 3800, 25108, 24282, 5924 |
| P06 | special | en | When is the Zorvian Lantern Festival held? | 3350, 3800 | 1 | 3350, 3800, 5924, 24282, 25108 |
| P07 | collection | en | The Saskatchewan Financial Services Commission issued a temporary cease trade order concerning diamond holdings in the Fort a la Corne area. Which city is that area east of? | 2143 | 2 | 2132, 2143, 2068, 2134, 2089 |
| P08 | collection | en | On the Alpha Fighting Systems upcoming events page, which Georgia city hosted the February 2009 Lutador Open? | 6008 | 1 | 6008, 30365, 8999, 22855, 14118 |
| P09 | collection | en | What is the name of the federated search engine described in ASM's December 2008 article about the mobile directory service launched in Singapore? | 10914 | 1 | 10914, 10891, 10894, 10929, 10933 |
| P10 | collection | en | On what date did the Stickam Blog announce Fashion Enemy welcoming Jerrell Scott from Project Runway? | 23077 | 5 | 23059, 23101, 23085, 23092, 23077 |
| P11 | collection | en | Which beach resort in Boca Chica was Treasure Divers located at, according to the Boca Chica Dominican Republic diving page? | 24640 | 1 | 24640, 24637, 24643, 24641, 24638 |
| P12 | collection | en | According to the San Diego onsite AutoCAD and Revit training schedule, on which date did Blueprint Reading 1 begin? | 27690 | 1 | 27690, 27685, 27675, 27681, 27673 |
| P13 | collection | en | On the ccMixter reviews page for The Secret Anger of Atoms, what is the title of the sample credited to greg_baumont? | 29865 | 1 | 29865, 29630, 29934, 29736, 29770 |
| P14 | collection | en | In Tam-Kien Duong's cendres.net post about in_place_edit and Rails 2, what plugin does the script/plugin install command install? | 30075 | 1 | 30075, 30072, 30073, 30071, 30074 |
| P15 | collection | en | What statewide exam-ranking cutoff got California campuses invited into the II/USP turnaround initiative, according to the overview from ALS? | 2580 | 1 | 2580, 24513, 24514, 3023, 24516 |
| P16 | collection | en | What street address does the Houston Press give for the conservancy behind the downtown park built on former parking lots that won its 2008 city-improvement award? | 11071 | 1 | 11071, 11058, 11054, 11066, 11114 |
| P17 | collection | zh | 阿拉斯加州公園介紹頁提到，威爾迪茲到費爾班克斯的古道上，哪間旅店在 1909 至 1947 年間接待旅客？ | 4507 | - | 31480, 31451, 31327, 31447, 31362 |
| P18 | collection | zh | 武具商店販售的二〇〇九年限量手繪日本書法月曆，頁面標示的美元售價是多少？ | 26078 | - | 31480, 31362, 31468, 31451, 31414 |
