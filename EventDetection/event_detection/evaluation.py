import collections
import os
import pickle

REAL_EVENTS = [
    (0, 152, '2. června – španělský král Juan Carlos I. abdikoval a za svého nástupce určil svého syna Filipa.'),
    (1, 154, '4. a 5. června – konal se 40. summit G8 v Bruselu.'),
    (2, 157, '7. června – Petro Porošenko složil prezidentskou přísahu a stal se prezidentem Ukrajiny.'),
    (3, 158, '8. června – Abd al-Fattáh as-Sísí složil prezidentskou přísahu a stal se prezidentem Egypta.'),
    (4, 160, "10. června – V izraelských prezidentských volbách byl zvolen Re'uven Rivlin"),
    (5, 162, '12. června – v Brazílii začalo 20. mistrovství světa ve fotbale.'),
    (6, 165, '15. června – Andrej Kiska složil prezidentskou přísahu a stal se prezidentem Slovenska.'),
    (7, 168,
     '18. června – vůdci vojenského převratu v Turecku z roku 1980 Kenan Evren a Tahsin Şahinkaya byli odsouzeni na doživotí.'),
    (8, 169, '19. června – Filip, asturský kníže složil přísahu a stal se králem Španělska jako Filip VI. Španělský'),
    (9, 181, '1. července – Itálie se ujala předsednictví EU.'),
    (10, 187,
     '8. července – armáda České republiky utrpěla největší ztrátu v novodobých dějinách, kdy při sebevražedném útoku poblíž letecké základny Bagram zemřeli čtyři čeští vojáci, spolu s dalšími 12 tamními oběťmi. Pátý český voják byl těžce raněn a 14. července zemřel.'),
    (11, 193, '13. července – mistry světa ve fotbale se stala německá fotbalová reprezentace.'),
    (12, 195,
     '15. července – novým předsedou Evropské komise se stal lucemburský politik a bývalý premiér Jean-Claude Juncker.'),
    (13, 197,
     '17. července – v oblasti bojů na východní Ukrajině se zřítil Boeing 777 malajsijských aerolinií. Zemřelo všech 295 osob na palubě.'),
    (14, 201,
     '21. července – vláda Bohuslava Sobotky vybrala nového eurokomisaře. Stane se jím ministryně pro místní rozvoj Věra Jourová, která ve výběru porazila Pavla Mertlíka.'),
    (15, 221,
     '10. srpna – V historicky první přímé prezidentské volbě v Turecku byl zvolen premiér Recep Tayyip Erdoğan.'),
    (16, 227, '16. srpna až 28. srpna – Letní olympijské hry mládeže 2014 v čínském Nankingu'),
    (17, 230,
     '19. srpna – americký novinář James Foley byl popraven v syrské poušti neznámým islámským radikálem, jeho smrt vyvolala v západním světe vlnu pobouření.'),
    (18, 235, '24. srpna – Meziplanetární sonda New Horizons prolétla blízko L5 soustavy Slunce–Neptun.'),
    (19, 236,
     '25. srpna – ve sporu o amnestii Václava Klause soud schválil smír, podle něhož se bývalý hradní právník Pavel Hasenkopf na vyhlášeném znění amnestie nepodílel.'),
    (20, 239, '28. srpna – Recep Tayyip Erdoğan složil prezidentskou přísahu a stal se prezidentem Turecka.'),
    (21, 241, '30. srpna – polský premiér Donald Tusk byl na summitu Evropské unie zvolen předsedou Evropské rady.'),
    (22, 243,
     '1. září – Pavel Hasenkopf podal na Vratislava Mynáře trestní oznámení pro pomluvu ohledně Mynářova výroku, že Hasenkopf je jedním z autorů amnestie Václava Klause.'),
    (23, 244,
     '2. září – další americký novinář Steven Sotloff byl popraven v syrské poušti neznámým islámským radikálem, stejně jako James Foley v srpnu.'),
    (24, 246,
     '4. září – ve Vilémově se zřítil most, na kterém probíhala rekonstrukce. Zemřeli čtyři dělníci, další dva byli zraněni.'),
    (25, 248, '6. září – počet nakažených ebolou při celoroční epidemii se přehoupl přes 4 000'),
    (26, 250, '8. září – britský následník trůnu Princ William a jeho manželka Kate oznámili, že čekají druhé dítě.'),
    (27, 252,
     '10. září – kandidátka na českou eurokomisařku Věra Jourová získala portfolio spravedlnosti, spotřebitelské politiky a rovnosti pohlaví.'),
    (28, 255,
     '13. září – islámští radikálové popravili dalšího západního zajatce, tentokrát jím byl britský humanitární pracovník David Haines.'),
    (29, 260,
     '18. září – Ve Skotsku proběhlo referendum o nezávislosti na Spojeném království. Pro odtržení od Británie hlasovalo 44,7% lidí, proti 55,3% lidí, Skotsko tak zůstane její součástí.'),
    (
        30, 262,
        '20. září – náčelník Generálního štábu Armády ČR Petr Pavel byl zvolen předsedou vojenského výboru NATO.'),
    (31, 266,
     '24. září – na Pražský hrad se dostal výhružný dopis adresovaný prezidentovi Miloši Zemanovi s bílým práškem. Případ šetří policie.'),
    (32, 275, '3. října – prezident Miloš Zeman přijal demisi ministryně pro místní rozvoj Věry Jourové.'),
    (33, 275,
     '3. října – islámští radikálové popravili dalšího západního zajatce, stal se jím opět britský humanitární pracovník Alan Henning.'),
    (34, 279,
     '7. října – evropský parlament schválil nominaci Věry Jourové na post eurokomisařky pro spravedlnost, spotřebitelskou politiku a rovnost pohlaví.'),
    (35, 282,
     '10. října a 11. října – proběhly volby do Senátu Parlamentu České republiky, volby do zastupitelstev obcí a volby do Zastupitelstva hlavního města Prahy. Ve volbách uspěly především vládní strany ČSSD, ANO a KDU-ČSL.'),
    (36, 286,
     '14. října – žena trpící schizofrenii pobodala na obchodní akademii ve Žďáru nad Sázavou tři studenty a zasahujícího policistu. Jeden ze studentů útok nepřežil.'),
    (37, 288,
     '16. října – Ve Vrběticích došlo k výbuchu muničního skladu č. 16. Na místě zahynuli dva zaměstnanci skladu, došlo k evakuaci obyvatel přilehlých obcí.'),
    (38, 288,
     '16. října – zanikla europarlamentní frakce Evropa svobody a přímé demokracie, 20. října byla opět obnovena.'),
    (39, 289,
     '17. října a 18. října – proběhlo druhé kolo voleb do Senátu Parlamentu České republiky. Ve volbách uspěly především vládní strany ČSSD, ANO a KDU-ČSL.'),
    (40, 312, '9. listopadu – v Katalánsku začalo symbolické hlasování o nezávislosti na Španělsku.'),
    (41, 315, '12. listopadu – přistál modul Philae jako historicky první lidský stroj na kometě.'),
    (42, 318,
     '15. listopad islámští radikálové popravili dalšího západního zajatce, stal se jím americký humanitární pracovník Peter Kassig.Komunální volby na Slovensku'),
    (43, 318, '15. a 16. listopadu – Summit G20 v Brisbane'),
    (44, 334,
     '1. prosince – ledovková kalamita ochromila hromadnou dopravu v ČR a dodávky elektřiny v mnoha regionech. Tramvajová doprava v Praze dokonce poprvé ve své historii zažila úplné zastavení provozu. Do normálu se dopravní i energetická situace vrátila až 3. prosince.'),
    (45, 334, '1. prosince – Druhým předsedou Evropské rady se stal Donald Tusk.'),
    (46, 336,
     '3. prosince – Ve Vrběticích došlo k dalšímu výbuchu muničního skladu č. 12. Opět proběhla evakuace obyvatel přilehlých obcí, oba dva výbuchy jsou vyšetřovány jako úmyslný trestný čin.    '),
    (47, 349,
     '16. prosince – Ozbrojenci ze skupiny Tahrík-e Tálibán-e Pákistán spáchali masakr v péšávarské vojenské škole škole. Útok si vyžádal 141 obětí většinu z nich tvořili děti.'),
    (48, 361,
     '28. prosince – na cestě ze Surabaje do Singapuru se ztratilo letadlo malajsijské společnosti AirAsia se 162 lidmi na palubě.')
]

# For Precision, Recall and F-measure. The format is (method, dps_boundary, mapping) with mapping being
# (detected_event_id, some_real_event_id), some_real_event_id = -1 if no such event is found.
EVENTS_PRF = [
    # ('clusters', 0.05, [(0, -1), (1, -1), (2, -1), (3, -1), (4, -1), (5, 13), (6, -1), (7, -1), (8, -1), (9, -1),
    #                     (10, -1), (11, -1), (12, 13), (13, -1), (14, -1), (15, -1), (16, -1), (17, 47), (18, -1),
    #                     (19, -1), (20, -1), (21, -1), (22, 5), (23, -1), (24, -1), (25, -1), (26, -1), (27, 44),
    #                     (28, 39)]),
    # ('clusters', 0.04, [(0, -1), (1, -1), (2, -1), (3, -1), (4, -1), (5, 17), (6, -1), (7, -1), (8, 39), (9, -1),
    #                     (10, 13), (11, -1), (12, -1), (13, -1), (14, -1), (15, -1), (16, -1), (17, 48), (18, -1),
    #                     (19, -1), (20, -1), (21, -1), (22, 5), (23, 47), (24, -1), (25, -1), (26, -1), (27, -1),
    #                     (28, -1), (29, 5), (30, -1), (31, -1), (32, -1), (33, -1), (34, -1), (35, 44), (36, 39)]),
    # ('clusters', 0.03, [(0, -1), (1, -1), (2, -1), (3, 5), (4, -1), (5, -1), (6, 25), (7, 10), (8, 17), (9, -1),
    #                     (10, -1), (11, 39), (12, -1), (13, 13), (14, -1), (15, -1), (16, -1), (17, -1), (18, -1),
    #                     (19, -1), (20, -1), (21, 46), (22, 13), (23, -1), (24, -1), (25, -1), (26, -1), (27, 5),
    #                     (28, 47), (29, -1), (30, -1), (31, -1), (32, -1), (33, -1), (34, -1), (35, 5), (36, -1),
    #                     (37, -1), (38, -1), (39, -1), (40, -1), (41, 44), (42, 39)]),
    # ('clusters', 0.02, [(0, -1), (1, -1), (2, -1), (3, -1), (4, 5), (5, -1), (6, -1), (7, 25), (8, 10), (9, -1),
    #                     (10, 17), (11, -1), (12, -1), (13, 35), (14, 35), (15, -1), (16, 13), (17, -1), (18, -1),
    #                     (19, -1), (20, -1), (21, -1), (22, -1), (23, -1), (24, -1), (25, 46), (26, 13), (27, -1),
    #                     (28, -1), (29, -1), (30, -1), (31, -1), (32, 5), (33, 47), (34, -1), (35, -1), (36, -1),
    #                     (37, 13), (38, -1), (39, -1), (40, -1), (41, -1), (42, 5), (43, -1), (44, -1), (45, -1),
    #                     (46, -1), (47, -1), (48, -1), (49, 44), (50, 35)]),
    ('clusters', 0.01, [(0, -1), (1, 35), (2, -1), (3, -1), (4, -1), (5, -1), (6, 10), (7, -1), (8, -1), (9, 5),
                        (10, -1), (11, -1), (12, -1), (13, -1), (14, 25), (15, -1), (16, 17), (17, -1), (18, -1),
                        (19, -1), (20, 35), (21, -1), (22, -1), (23, -1), (24, -1), (25, 13), (26, 13), (27, -1),
                        (28, -1), (29, -1), (30, -1), (31, -1), (32, -1), (33, -1), (34, -1), (35, -1), (36, -1),
                        (37, 29), (38, -1), (39, -1), (40, -1), (41, -1), (42, -1), (43, 46), (44, 13), (45, -1),
                        (46, -1), (47, -1), (48, -1), (49, -1), (50, -1), (51, -1), (52, 5), (53, -1), (54, -1),
                        (55, -1), (56, 44), (57, 13), (58, -1), (59, -1), (60, -1), (61, -1), (62, 5), (63, 25),
                        (64, 39), (65, -1), (66, -1), (67, -1), (68, 35), (69, -1), (70, -1), (71, -1), (72, -1),
                        (73, -1), (74, -1), (75, 44), (76, 39)]),
    # ('clusters-separately', 0.05, [(0, -1), (1, -1), (2, -1), (3, 17), (4, -1), (5, -1), (6, -1), (7, -1),
    #                                (8, -1), (9, -1), (10, -1), (11, -1), (12, -1), (13, -1), (14, 5), (15, -1),
    #                                (16, -1), (17, -1), (18, -1), (19, 44), (20, -1), (21, -1), (22, 17), (23, 13),
    #                                (24, -1), (25, -1), (26, 13), (27, -1), (28, 47), (29, -1), (30, -1), (31, -1),
    #                                (32, 35)]),
    # ('clusters-separately', 0.04, [(0, -1), (1, -1), (2, -1), (3, 17), (4, -1), (5, 17), (6, -1), (7, -1),
    #                                (8, -1), (9, -1), (10, -1), (11, -1), (12, -1), (13, -1), (14, -1), (15, -1),
    #                                (16, -1), (17, -1), (18, -1), (19, 5), (20, -1), (21, -1), (22, -1), (23, 44),
    #                                (24, -1), (25, -1), (26, 17), (27, 35), (28, 13), (29, -1), (30, -1), (31, 13),
    #                                (32, 5), (33, 47), (34, -1), (35, -1), (36, -1), (37, -1), (38, -1), (39, -1),
    #                                (40, 35)]),
    # ('greedy', 0.05, [(0, 39), (1, -1), (2, -1), (3, -1), (4, -1), (5, -1), (6, -1), (7, -1), (8, -1), (9, -1),
    #                   (10, 35), (11, -1), (12, -1), (13, -1), (14, -1), (15, -1), (16, -1), (17, -1), (18, -1),
    #                   (19, -1), (20, -1), (21, -1), (22, 13), (23, 44), (24, -1), (25, -1), (26, 13), (27, 5),
    #                   (28, -1), (29, -1), (30, -1), (31, -1), (32, -1), (33, -1), (34, -1), (35, -1), (36, -1),
    #                   (37, -1), (38, 17), (39, -1), (40, -1), (41, -1), (42, 47), (43, -1), (44, -1), (45, -1),
    #                   (46, -1), (47, -1), (48, -1), (49, -1), (50, -1), (51, -1), (52, -1), (53, -1), (54, -1),
    #                   (55, 46), (56, -1), (57, -1), (58, -1), (59, -1), (60, -1), (61, -1), (62, -1), (63, 39),
    #                   (64, 39), (65, -1), (66, -1), (67, -1), (68, 47), (69, 30), (70, -1)]),
    ('greedy', 0.04, [(0, -1), (1, -1), (2, -1), (3, -1), (4, -1), (5, -1), (6, -1), (7, -1), (8, 35), (9, -1),
                      (10, -1), (11, -1), (12, -1), (13, -1), (14, -1), (15, -1), (16, -1), (17, -1), (18, -1),
                      (19, -1), (20, -1), (21, -1), (22, -1), (23, -1), (24, -1), (25, -1), (26, -1), (27, -1),
                      (28, -1), (29, -1), (30, -1), (31, -1), (32, -1), (33, -1), (34, -1), (35, -1), (36, -1),
                      (37, -1), (38, -1), (39, -1), (40, -1), (41, -1), (42, -1), (43, -1), (44, -1), (45, -1),
                      (46, 5), (47, -1), (48, -1), (49, -1), (50, -1), (51, 5), (52, -1), (53, -1), (54, 10),
                      (55, -1), (56, 35), (57, 13), (58, -1), (59, -1), (60, 47), (61, -1), (62, -1), (63, 13),
                      (64, -1), (65, -1), (66, 35), (67, -1), (68, 13), (69, -1), (70, 35), (71, 1), (72, -1),
                      (73, 35), (74, 47), (75, 13), (76, 35), (77, -1), (78, 47), (79, -1), (80, -1), (81, -1)]),
    # ('greedy', 0.03, [(0, -1), (1, -1), (2, -1), (3, -1), (4, -1), (5, -1), (6, 17), (7, -1), (8, -1), (9, -1),
    #                   (10, -1), (11, -1), (12, -1), (13, -1), (14, -1), (15, -1), (16, -1), (17, 44), (18, -1),
    #                   (19, -1), (20, 35), (21, -1), (22, -1), (23, -1), (24, -1), (25, -1), (26, -1), (27, -1),
    #                   (28, -1), (29, 35), (30, -1), (31, -1), (32, -1), (33, -1), (34, -1), (35, -1), (36, -1),
    #                   (37, -1), (38, -1), (39, -1), (40, -1), (41, 47), (42, -1), (43, -1), (44, -1), (45, -1),
    #                   (46, -1), (47, -1), (48, -1), (49, 5), (50, -1), (51, -1), (52, -1), (53, -1), (54, -1),
    #                   (55, 13), (56, -1), (57, -1), (58, 17), (59, -1), (60, -1), (61, -1), (62, -1), (63, -1),
    #                   (64, -1), (65, -1), (66, -1), (67, 35), (68, -1), (69, -1), (70, -1), (71, -1), (72, -1),
    #                   (73, 35), (74, -1), (75, -1), (76, -1), (77, -1), (78, -1), (79, -1), (80, -1), (81, -1),
    #                   (82, -1), (83, 35), (84, -1), (85, -1), (86, -1), (87, -1), (88, 47), (89, 47), (90, -1),
    #                   (91, -1), (92, -1), (93, -1), (94, -1), (95, 35), (96, -1), (97, -1), (98, -1), (99, -1),
    #                   (100, -1), (101, -1), (102, -1), (103, -1), (104, -1)]),
    # ('greedy-separately', 0.05, [(0, -1), (1, -1), (2, -1), (3, -1), (4, -1), (5, -1), (6, -1), (7, -1), (8, 17),
    #                              (9, -1), (10, 44), (11, -1), (12, -1), (13, -1), (14, -1), (15, -1), (16, -1),
    #                              (17, -1), (18, -1), (19, -1), (20, 47), (21, -1), (22, 35), (23, -1), (24, -1),
    #                              (25, -1), (26, -1), (27, -1), (28, -1), (29, -1), (30, -1), (31, -1), (32, -1),
    #                              (33, -1), (34, -1), (35, -1), (36, 35), (37, -1), (38, -1), (39, -1), (40, -1),
    #                              (41, -1), (42, -1), (43, -1), (44, -1), (45, -1), (46, 13), (47, -1), (48, 13),
    #                              (49, -1), (50, -1), (51, -1), (52, -1), (53, 35), (54, -1), (55, -1), (56, -1),
    #                              (57, -1), (58, 35), (59, -1), (60, -1), (61, -1), (62, -1), (63, -1), (64, -1),
    #                              (65, 47), (66, 30), (67, -1)]),
    ('original', 0.05, [(0, -1), (1, -1), (2, -1), (3, -1), (4, -1), (5, -1), (6, -1), (7, -1), (8, -1), (9, -1),
                        (10, -1), (11, -1), (12, -1), (13, -1), (14, -1), (15, 17), (16, -1), (17, -1), (18, -1),
                        (19, 47), (20, -1), (21, -1), (22, -1), (23, -1), (24, -1), (25, -1), (26, -1), (27, -1),
                        (28, -1), (29, -1), (30, -1), (31, -1), (32, -1), (33, -1), (34, 23), (35, -1), (36, -1),
                        (37, -1), (38, -1), (39, -1), (40, -1), (41, -1), (42, 5), (43, -1), (44, -1), (45, -1),
                        (46, -1), (47, -1), (48, -1), (49, -1), (50, -1), (51, 35), (52, -1), (53, 23), (54, -1),
                        (55, -1), (56, -1), (57, -1), (58, -1), (59, -1), (60, -1), (61, -1), (62, -1), (63, -1),
                        (64, -1), (65, -1), (66, -1), (67, -1), (68, -1), (69, -1), (70, -1), (71, -1), (72, 35),
                        (73, -1), (74, -1), (75, -1), (76, -1), (77, 17), (78, 28), (79, -1), (80, -1), (81, -1),
                        (82, 13), (83, -1), (84, -1), (85, -1), (86, -1), (87, -1), (88, -1), (89, -1), (90, -1),
                        (91, -1), (92, -1), (93, -1), (94, -1), (95, -1), (96, -1), (97, -1), (98, -1), (99, 47),
                        (100, 23), (101, -1), (102, -1), (103, -1), (104, -1), (105, 44), (106, -1), (107, -1),
                        (108, -1), (109, 5), (110, 2), (111, -1), (112, -1), (113, -1), (114, -1), (115, -1),
                        (116, -1), (117, 35), (118, -1), (119, -1), (120, -1), (121, -1), (122, -1), (123, -1),
                        (124, 35), (125, -1), (126, 4), (127, -1), (128, -1), (129, 35), (130, -1), (131, -1),
                        (132, -1), (133, 47), (134, -1), (135, -1), (136, -1), (137, -1), (138, -1), (139, 47),
                        (140, -1), (141, -1), (142, 47), (143, 47), (144, -1), (145, -1), (146, -1), (147, -1),
                        (148, -1), (149, -1), (150, 30), (151, -1), (152, -1), (153, -1), (154, -1), (155, -1),
                        (156, 13), (157, -1), (158, 35)])]

CLUSTERS_PRECISION = [(0, -1), (1, 35), (2, -1), (3, -1), (4, -1), (5, -1), (6, 10), (7, -1), (8, -1), (9, 5),
                      (10, -1), (11, -1), (12, -1), (13, -1), (14, 25), (15, -1), (16, 17), (17, -1), (18, -1),
                      (19, -1), (20, 35), (21, -1), (22, -1), (23, -1), (24, -1), (25, 13), (26, 13), (27, -1),
                      (28, -1), (29, -1), (30, -1), (31, -1), (32, -1), (33, -1), (34, -1), (35, -1), (36, -1),
                      (37, 29), (38, -1), (39, -1), (40, -1), (41, -1), (42, -1), (43, 46), (44, 13), (45, -1),
                      (46, -1), (47, -1), (48, -1), (49, -1), (50, -1), (51, -1), (52, 5), (53, -1), (54, -1),
                      (55, -1), (56, 44), (57, 13), (58, -1), (59, -1), (60, -1), (61, -1), (62, 5), (63, 25),
                      (64, 39), (65, -1), (66, -1), (67, -1), (68, 35), (69, -1), (70, -1), (71, -1), (72, -1),
                      (73, -1), (74, -1), (75, 44), (76, 39)]
CLUSTERS_RECALL = [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0,
                   0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1]

GREEDY_PRECISION = [(0, -1), (1, -1), (2, -1), (3, -1), (4, -1), (5, -1), (6, -1), (7, -1), (8, 35), (9, -1),
                    (10, -1), (11, -1), (12, -1), (13, -1), (14, -1), (15, -1), (16, -1), (17, -1), (18, -1),
                    (19, -1), (20, -1), (21, -1), (22, -1), (23, -1), (24, -1), (25, -1), (26, -1), (27, -1),
                    (28, -1), (29, -1), (30, -1), (31, -1), (32, -1), (33, -1), (34, -1), (35, -1), (36, -1),
                    (37, -1), (38, -1), (39, -1), (40, -1), (41, -1), (42, -1), (43, -1), (44, -1), (45, -1),
                    (46, 5), (47, -1), (48, -1), (49, -1), (50, -1), (51, 5), (52, -1), (53, -1), (54, 10),
                    (55, -1), (56, 35), (57, 13), (58, -1), (59, -1), (60, 47), (61, -1), (62, -1), (63, 13),
                    (64, -1), (65, -1), (66, 35), (67, -1), (68, 13), (69, -1), (70, 35), (71, 1), (72, -1),
                    (73, 35), (74, 47), (75, 13), (76, 35), (77, -1), (78, 47), (79, -1), (80, -1), (81, -1)]
GREEDY_RECALL = [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1]

ORIGINAL_PRECISION = [(0, -1), (1, -1), (2, -1), (3, -1), (4, -1), (5, -1), (6, -1), (7, -1), (8, -1), (9, -1),
                      (10, -1), (11, -1), (12, -1), (13, -1), (14, -1), (15, 17), (16, -1), (17, -1), (18, -1),
                      (19, 47), (20, -1), (21, -1), (22, -1), (23, -1), (24, -1), (25, -1), (26, -1), (27, -1),
                      (28, -1), (29, -1), (30, -1), (31, -1), (32, -1), (33, -1), (34, 23), (35, -1), (36, -1),
                      (37, -1), (38, -1), (39, -1), (40, -1), (41, -1), (42, 5), (43, -1), (44, -1), (45, -1),
                      (46, -1), (47, -1), (48, -1), (49, -1), (50, -1), (51, 35), (52, -1), (53, 23), (54, -1),
                      (55, -1), (56, -1), (57, -1), (58, -1), (59, -1), (60, -1), (61, -1), (62, -1), (63, -1),
                      (64, -1), (65, -1), (66, -1), (67, -1), (68, -1), (69, -1), (70, -1), (71, -1), (72, 35),
                      (73, -1), (74, -1), (75, -1), (76, -1), (77, 17), (78, 28), (79, -1), (80, -1), (81, -1),
                      (82, 13), (83, -1), (84, -1), (85, -1), (86, -1), (87, -1), (88, -1), (89, -1), (90, -1),
                      (91, -1), (92, -1), (93, -1), (94, -1), (95, -1), (96, -1), (97, -1), (98, -1), (99, 47),
                      (100, 23), (101, -1), (102, -1), (103, -1), (104, -1), (105, 44), (106, -1), (107, -1),
                      (108, -1), (109, 5), (110, 2), (111, -1), (112, -1), (113, -1), (114, -1), (115, -1),
                      (116, -1), (117, 35), (118, -1), (119, -1), (120, -1), (121, -1), (122, -1), (123, -1),
                      (124, 35), (125, -1), (126, 4), (127, -1), (128, -1), (129, 35), (130, -1), (131, -1),
                      (132, -1), (133, 47), (134, -1), (135, -1), (136, -1), (137, -1), (138, -1), (139, 47),
                      (140, -1), (141, -1), (142, 47), (143, 47), (144, -1), (145, -1), (146, -1), (147, -1),
                      (148, -1), (149, -1), (150, 30), (151, -1), (152, -1), (153, -1), (154, -1), (155, -1),
                      (156, 13), (157, -1), (158, 35)]
ORIGINAL_RECALL = [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0,
                   0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1]

# Events grouped by relatedness.
EVENTS_REDUNDANCY = [
    # ('clusters', 0.05, [[0, 2, 7, 19], [1, 17], [3], [4], [5], [6], [8, 25], [9], [10], [11], [12], [13], [14],
    #                     [15, 26], [16, 18], [20], [21, 27], [22], [23], [24], [28]]),
    # ('clusters', 0.04, [[0, 2, 4, 11, 26], [1, 23], [3], [5], [6, 9], [7], [8, 36], [10], [12], [13, 32], [14], [15],
    #                     [16], [17], [18], [19], [20, 21, 33], [22], [24], [25], [27], [28, 35], [29], [30], [31],
    #                     [34]]),
    # ('clusters', 0.03, [[0, 1, 5, 14, 32], [4, 28], [11, 42], [16, 17, 38], [13], [22], [2], [3, 35], [6], [7], [8],
    #                     [9, 12], [10], [15], [18], [19], [20], [21], [23], [24], [25, 26, 39], [27, 30, 36], [29],
    #                     [31], [33], [34, 41], [37], [40]]),
    # ('clusters', 0.02, [[0, 2, 6, 17, 39], [5, 33], [13, 14, 50], [20, 21, 46], [16], [26, 37], [3], [4, 42], [7], [8],
    #                     [10], [11, 15], [12], [18], [22], [23], [24], [25], [27], [28], [19, 29, 30, 31, 47],
    #                     [32, 35, 44], [34], [36, 38], [40], [41, 49], [45], [48], [1], [9], [43]]),
    ('clusters', 0.01, [[0, 2, 4, 10, 12, 15, 28, 47, 59],
                        [11], [1, 20, 76], [34, 35, 69], [26, 38], [25, 44, 57], [7], [9, 62], [14, 63], [16], [6],
                        [19, 22, 24, 31, 64], [8], [29], [36], [40], [42], [43], [46], [48],
                        [27, 30, 49, 50, 51, 70, 72], [52, 54, 66], [53], [55, 58, 71, 74], [60], [61, 75], [67], [73],
                        [3], [17], [65], [5], [13], [18], [21], [23], [32], [33], [37], [39], [41], [45], [56], [68]]),
    ('greedy', 0.04, [[0, 25, 28, 29, 30, 31, 33, 35, 38, 41, 45, 47, 50, 55, 59, 67, 69, 71, 72, 80],
                      [1, 12, 13, 36, 44, 53, 54, 61], [2, 6, 19, 22, 60, 70],
                      [3, 4, 7, 16, 17, 20, 49, 57, 65, 74, 77, 79], [5, 15, 21], [8, 56, 66, 73, 76],
                      [9, 10, 14, 23, 40, 43], [11, 24], [18, 26, 37, 39, 58, 62], [27], [32], [34], [42], [46, 51],
                      [48], [52], [63], [64, 81], [68, 75], [78]]),
    # ('greedy-kl', 0.04, [[0, 1, 7, 10, 14, 18, 19, 22, 24, 25, 26, 29, 36, 39, 43, 45, 47, 50, 51, 52, 54, 55, 56, 57,
    #                       58], [2, 3, 4, 11, 12, 13, 17, 38, 46, 49, 53, 59], [5, 15, 37], [6], [8, 33], [9, 35],
    #                      [16, 21, 27, 31, 40], [20], [23, 60], [28], [30], [32], [34], [41], [42, 44], [48]]),
    ('original', 0.05, [[0, 6, 85],
                        [1, 2, 3, 4, 5, 9, 10, 13, 14, 17, 18, 20, 21, 24, 27, 28, 29, 30, 31, 33, 35, 37, 40, 41, 43,
                         45, 46, 49, 50, 54, 55, 57, 58, 59, 60, 63, 64, 65, 67, 68, 70, 71, 74, 76, 79, 80, 81, 83, 84,
                         86, 87, 88, 89, 90, 91, 93, 94, 98, 102, 106, 108, 118, 122, 123, 124, 127, 129, 130, 133, 134,
                         135, 137, 139, 140, 143, 144, 145, 146, 148, 150, 151, 156], [7, 8],
                        [11, 12, 23, 26, 39, 47, 66, 115], [15, 34, 77, 78, 100, 125], [16], [19, 99], [22], [25, 116],
                        [32, 92], [36, 97, 119, 155], [38], [42, 109], [44, 104], [48],
                        [51, 72, 116, 123, 128, 136, 157], [52, 53, 73, 95, 107], [56], [61], [62, 101, 103, 111, 114],
                        [69], [75], [82], [96], [105], [110], [112, 113], [120, 153, 154], [121, 126], [131, 147],
                        [132], [138], [141, 142], [149], [152]])
]

# Events grouped into (noisy, good).
EVENTS_NOISINESS = [
    # ('clusters', 0.03, ([14, 17, 25, 26, 27, 29, 30, 31, 32, 33, 36, 37, 39],
    #                     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 18, 19, 20, 21, 22, 23, 24, 28, 34, 35,
    #                      38, 40, 41, 42])),
    # ('clusters', 0.02, ([17, 19, 21, 29, 30, 31, 32, 34, 36, 38, 40, 43, 44, 45, 47],
    #                     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 23, 24, 25, 26, 27, 28,
    #                      33, 35, 37, 39, 41, 42, 46, 48, 49, 50])),
    ('clusters', 0.01, ([18, 21, 27, 30, 32, 49, 50, 51, 53, 58, 60, 67, 70, 72, 74],
                        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 22, 23, 24, 25, 26, 28,
                         29, 31, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 52, 54, 55, 56, 57, 59,
                         61, 62, 63, 64, 65, 66, 68, 69, 71, 73, 75, 76])),
    ('greedy', 0.04, ([1, 2, 3, 4, 6, 7, 12, 13, 16, 17, 19, 20, 22, 29, 36, 44, 49, 53, 54, 55, 57, 60, 61, 65, 70, 71,
                       74, 77, 79, 80, 81],
                      [0, 5, 8, 9, 10, 11, 14, 15, 18, 21, 23, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 35, 37, 38, 39,
                       40, 41, 42, 43, 45, 46, 47, 48, 50, 51, 52, 56, 58, 59, 62, 63, 64, 66, 67, 68, 69, 72, 73, 75,
                       76, 78])),
    # ('greedy-kl', 0.04, ([2, 3, 4, 8, 10, 11, 12, 13, 14, 17, 18, 28, 29, 31, 35, 36, 38, 44, 45, 46, 49, 50, 52, 53,
    #                       57, 59],
    #                      [0, 1, 5, 6, 7, 9, 15, 16, 19, 20, 21, 22, 23, 24, 25, 26, 27, 30, 32, 33, 34, 37, 39, 40, 41,
    #                       42, 43, 47, 48, 51, 54, 55, 56, 58, 60])),
    ('original', 0.05, ([0, 5, 6, 8, 12, 22, 23, 26, 38, 39, 47, 48, 51, 52, 53, 56, 60, 61, 62, 63, 64, 65, 66, 67, 68,
                         69, 70, 71, 72, 73, 76, 78, 80, 83, 84, 85, 87, 89, 90, 91, 92, 94, 95, 96, 97, 98, 99, 101,
                         102, 103, 105, 106, 108, 111, 118, 119, 121, 123, 125, 126, 127, 128, 130, 131, 132, 133, 134,
                         135, 136, 137, 138, 140, 141, 142, 144, 145, 146, 147, 149, 151, 154],
                        [1, 2, 3, 4, 7, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 27, 28, 29, 30, 31, 32,
                         33, 34, 35, 36, 37, 40, 41, 42, 43, 44, 45, 46, 49, 50, 54, 55, 57, 58, 59, 74, 75, 77, 79, 81,
                         82, 86, 88, 93, 100, 104, 107, 109, 110, 112, 113, 114, 115, 116, 117, 120, 122, 124, 129,
                         139, 143, 148, 150, 152, 153, 155, 156, 157, 158]))
]

CLUSTERS_EVENT_KEYWORDS = [
    ['Adaptive', 'Direct', 'Eija', 'Golf', 'Hama', 'Harmonious', 'Infiniti', 'Japonec', 'Leon', 'Lexus', 'MQB', 'NASA',
     'Network', 'Nissan', 'Nunziola', 'Octavio', 'Quanto', 'Road', 'Seat', 'Steering', 'Toyoda', 'VTEC', 'Vechia',
     'akumulátor', 'carsharing', 'charakter', 'citlivý', 'dobít', 'dojezd', 'elektrolyt', 'elektromotor',
     'futuristický', 'hatchback', 'image', 'instalovaný', 'ionový', 'jízdní', 'kabel', 'kapota', 'karosérie',
     'klientela', 'komfort', 'kompakt', 'kompaktní', 'koncernový', 'krytý', 'lichtenštejnský', 'lukrativní', 'mapující',
     'mobilita', 'mokrý', 'navazovat', 'náboj', 'odbytiště', 'onout', 'patentovaný', 'platforma', 'podporující',
     'pojem', 'posouvající', 'pozoruhodný', 'pozvolna', 'pracující', 'praktický', 'pětidveřový', 'přenášet',
     'připomínající', 'příkaz', 'přínos', 'příď', 'realizace', 'sedan', 'sedící', 'shodný', 'sklo', 'spolehlivost',
     'studie', 'svažující', 'trend', 'tříkolka', 'udávaný', 'ujet', 'volant', 'vpředu', 'vtrhnout', 'vyklápěcí',
     'využívající', 'využívání', 'vzhled', 'vážit', 'vážící', 'výzkumný', 'wire', 'zbytný', 'značný', 'úhel', 'čelní',
     'čtyřsedadlový'],
    ['Adriano', 'Krnáčová', 'primátorka'],
    ['Alta', 'Celerio', 'funkčnost', 'rozvora', 'sázející', 'udávat', 'užitný'],
    ['Andrej', 'Babiš'],
    ['Audi', 'Bosch', 'Championship', 'Citroen', 'Civic', 'Climawork', 'DOHC', 'Daimler', 'ETH', 'Endurance', 'Energy',
     'Honda', 'Mans', 'Peugeot', 'Pionýr', 'Porsche', 'PureTec', 'Renault', 'THP', 'Tourer', 'Toyota', 'Turba', 'Twin',
     'World', 'agregát', 'ambice', 'atmosférický', 'automobil', 'automobilismus', 'automobilka', 'automobilový',
     'bavorský', 'benzínový', 'brzdový', 'celohliníkový', 'centimetr', 'charakterizovat', 'chod', 'citroen', 'curyšský',
     'dCe', 'diesel', 'dohánět', 'dokola', 'doplnění', 'dotyk', 'downsizing', 'drahý', 'drát', 'dveřový', 'dvojice',
     'dvoulitrový', 'dynamický', 'délka', 'ekologický', 'euforie', 'exkluzivní', 'fáze', 'hnací', 'hodit', 'hodlat',
     'hospodárný', 'hrana', 'hybrid', 'hybridní', 'infrastruktura', 'inovativní', 'kategorie', 'klasický', 'kolej',
     'kombík', 'komercializovat', 'komplexní', 'koncern', 'kupé', 'laboratoř', 'laminární', 'litr', 'luxus', 'maxima',
     'mezitím', 'množství', 'model', 'modelový', 'modernizace', 'moment', 'motor', 'myšlenka', 'naftový', 'nakládání',
     'nanometr', 'nastartovat', 'norma', 'obdobný', 'oceňovat', 'odpovídat', 'odstavit', 'odvodit', 'odvážný', 'otálet',
     'označení', 'palivo', 'palivový', 'parametr', 'pedál', 'perfektně', 'plnění', 'podání', 'pohon', 'poněkud',
     'pospíšit', 'posunout', 'posunovat', 'posunutý', 'potrubí', 'povinný', 'prezentovat', 'prius', 'prodloužený',
     'prodlužovat', 'prostorný', 'proudění', 'provedený', 'průběžně', 'předpis', 'předpokládat', 'přelomový',
     'přeplňovaný', 'přicházet', 'přinášet', 'přinést', 'realistický', 'redukce', 'reprezentovat', 'revoluční',
     'rozumný', 'rozvod', 'rozvíjet', 'rozšířený', 'ryze', 'sací', 'sen', 'skleníkový', 'složitý', 'smysl', 'snižovat',
     'součinitel', 'spin', 'splnit', 'splňovat', 'spojka', 'sportovní', 'spotřeba', 'standardní', 'stopa', 'strategie',
     'stuttgartský', 'střední', 'sériový', 'takto', 'technický', 'technologie', 'terén', 'točivý', 'turba',
     'turbodmychadlo', 'tvarování', 'typický', 'třída', 'tříválec', 'udržovat', 'vedlejší', 'velikost', 'verze',
     'vodíkový', 'vrcholný', 'vstřik', 'vybudování', 'vyhledat', 'vylepšit', 'vypínat', 'vyrobený', 'vytrvalostní',
     'vyzkoušet', 'vyšlápnutí', 'vzduch', 'vzniknout', 'výbava', 'výběr', 'výfuk', 'výstižně', 'vývojový', 'vůz',
     'zachování', 'zajímavý', 'zaměřit', 'zapsat', 'zastavení', 'zavazadelník', 'zbavit', 'zdokonalovat', 'zkorigovaný',
     'značka', 'zpozornět', 'zpoždění', 'zredukovat', 'zveřejněný', 'zvýraznění', 'zvětšený', 'záležet', 'záď',
     'získaný', 'úplný', 'úsporný', 'čerpací', 'čtyřdveřový', 'čtyřválec', 'šarm'],
    ['Australian', 'Melbourne'],
    ['Bagdád', 'Irák', 'irácký'],
    ['Bartošová', 'Iveta', 'Rychtář'],
    ['Betlém', 'Ježíšek', 'Vánoce', 'advent', 'adventní', 'betlém', 'cukroví', 'dárek', 'kapr', 'koleda', 'ozdoba',
     'prosinec', 'předvánoční', 'stromek', 'stromeček', 'svátek', 'sváteční', 'vánoční', 'štědrovečerní', 'štědrý'],
    ['Brazílie', 'brazilský'],
    ['Brémy', 'CNG', 'Citigo', 'Combi', 'Corporation', 'DIG', 'Designwork', 'Ecoboost', 'Electric', 'Focus', 'GTC',
     'Gasoline', 'Kalifornie', 'LUV', 'Mercedes-Benz', 'Mitsubishi', 'Motors', 'Octavia', 'Outlander', 'PHEV',
     'Qashqai', 'Stuttgart', 'Sunnyvale', 'TEC', 'Tesla', 'analogicky', 'carport', 'chvála', 'dobíjení', 'drive',
     'estét', 'inženýr', 'jumper', 'kalifornský', 'karosářský', 'kombi', 'koncepční', 'konvenční', 'kúra', 'kříženec',
     'minivůz', 'modernizovaný', 'modernizovat', 'natankovaný', 'netřeba', 'odklon', 'odstartování', 'odtučňovací',
     'paleta', 'paralela', 'partnerský', 'patřící', 'plnička', 'plynule', 'poháněný', 'počínající', 'pás', 'pět',
     'předepsaný', 'předvádění', 'přeměna', 'přestěhovat', 'přizpůsobení', 'rozrůst', 'sestavení', 'sjíždět',
     'sluneční', 'solární', 'sourozenec', 'stlačený', 'sympózium', 'technolog', 'testovací', 'vizuální', 'vyvolaný',
     'váznoucí', 'vídeňský', 'wallbox', 'zodpovídat', 'řídicí'],
    ['Charlie', 'Hebdo', 'Mohamed', 'Paříž', 'islám', 'karikatura', 'muslim', 'muslimský', 'náboženství', 'pařížský',
     'prorok', 'satirický', 'teroristický', 'terorizmus'],
    ['China', 'Cross', 'Ostřihom', 'Peking', 'SUV', 'Suzuki', 'Transporter', 'Volkswagen', 'absolutní', 'autosalón',
     'brána', 'crossover', 'decentně', 'dno', 'dveřní', 'dvouciferný', 'extrémní', 'konkrétně', 'ližina', 'lákat',
     'maďarský', 'mimořádně', 'navýšený', 'nazutý', 'náprava', 'nárůst', 'okno', 'osudový', 'otevření', 'pevný',
     'plast', 'pneumatika', 'podběh', 'pomalu', 'prototyp', 'práh', 'punc', 'převis', 'připojovat', 'repertoár',
     'ručně', 'salón', 'sloupec', 'snový', 'stálý', 'stěhovat', 'střed', 'střešní', 'světlost', 'terénní',
     'trojúhelníkový', 'užitkový', 'velkolepý', 'vyrábět', 'výhled', 'zadní', 'zorný', 'zvedat', 'zúčastnit', 'říše'],
    ['Dakar', 'Loprais'],
    ['Ebola', 'Guinea', 'Leone', 'Libérie', 'Sierra', 'ebola', 'epidemie'],
    ['Ford', 'elektromobil'],
    ['Gaza', 'Hamas', 'Izrael', 'Izraelec', 'Palestinec', 'izraelský', 'palestinský'],
    ['Hamilton', 'Rosberg', 'Vettel'],
    ['Hradec', 'Králové'],
    ['Janukovyč', 'Janukovyčův', 'Viktor'],
    ['KDU-ČSL', 'lidovec'],
    ['Karlův', 'Vary'],
    ['Klička', 'Vitalij'],
    ['Kostarika', 'Uruguay'],
    ['Krym', 'Sevastopol', 'krymský', 'poloostrov'],
    ['Kuala', 'Lumpur'],
    ['Kyjev', 'Ukrajina', 'ukrajinský'],
    ['Ltd', 'Telecom', 'hyperodkaz', 'obsáhnout', 'rádiový', 'stack'],
    ['Mercedes', 'mercedes'],
    ['Miloš', 'Zeman', 'Zemanův'],
    ['OSKB', 'kolektivní', 'pakt', 'přiblížení', 'symetricky'],
    ['Oleksandr', 'Turčynov'],
    ['Plzeň', 'plzeň'],
    ['Pussy', 'Riot'],
    ['Putin', 'Vladimir'],
    ['Rusko', 'ruský'],
    ['Silvestr', 'novoroční', 'ohňostroj', 'silvestr', 'silvestrovský'],
    ['Skotsko', 'skotský'],
    ['Slavjansk', 'Slavjansko'],
    ['Soukalová', 'biatlonistka'],
    ['Soča', 'Soči', 'ZOH', 'olympijský', 'olympiáda'],
    ['Třinec', 'ocelář'],
    ['Velikonoce', 'velikonoční'],
    ['Vrbětice', 'muniční', 'sklad'],
    ['boeing', 'malajsijský', 'mha', 'sestřelení', 'sestřelený'],
    ['bouřka', 'přívalový'],
    ['branka', 'brankář', 'dneska', 'dobře', 'docela', 'dost', 'duel', 'dělat', 'gól', 'gólman', 'hned', 'host',
     'hrát', 'minuta', 'moc', 'možná', 'nakonec', 'neděle', 'nedělní', 'někdy', 'obránce', 'opravdu', 'porazit',
     'porážka', 'prohra', 'prohrát', 'prostě', 'pátek', 'přihrávka', 'radovat', 'rád', 'sezona', 'sezóna', 'skóre',
     'skórovat', 'snad', 'sobota', 'sobotní', 'souboj', 'soupeř', 'střela', 'tabulka', 'takhle', 'taky', 'tam',
     'tentokrát', 'teď', 'trefa', 'trefit', 'triumf', 'trochu', 'tyč', 'tým', 'udělat', 'utkání', 'vstřelit',
     'vyhrát', 'vyrovnat', 'vítězství', 'výhra', 'vědět', 'vždycky', 'zase', 'zdolat', 'zvítězit', 'zápas', 'úplně'],
    ['céčko', 'vymřít'],
    ['dovolená', 'letní', 'léto', 'prázdninový', 'prázdniny'],
    ['edce', 'rhhar', 'service', 'tohi', 'uejt'],
    ['extra', 'podnik', 'reklama', 'siga', 'zpravodajství'],
    ['formátování', 'grafický', 'grafik', 'grafika', 'móda', 'pokročilý', 'prohlížeč', 'přepnout', 'zdobný'],
    ['fotbalista', 'fotbalový'],
    ['kliknout', 'předplatitelský'],
    ['kop', 'pokutový', 'vápno'],
    ['kouč', 'trenér'],
    ['ledovka', 'námraza'],
    ['letadlo', 'letoun'],
    ['liga', 'ligový'],
    ['litrový', 'plug-in'],
    ['loňský', 'vloni'],
    ['lyže', 'lyžování'],
    ['mistrovství', 'šampionát'],
    ['nakazit', 'nakažený'],
    ['opozice', 'opoziční'],
    ['ples', 'pleso'],
    ['poločas', 'půle'],
    ['pondělní', 'pondělí', 'středa', 'úterní', 'úterý', 'čtvrtek'],
    ['poslanecký', 'sněmovna'],
    ['proruský', 'separatista'],
    ['předplatit', 'předplatné'],
    ['přesilovka', 'přesilový'],
    ['read', 'string', 'val'],
    ['revoluce', 'sametový'],
    ['sekunda', 'vteřina'],
    ['sníh', 'sněhový'],
    ['volba', 'volební', 'volič']
]

CLUSTERS_EVENT_BURSTS = [
    [(141, 163)], [(276, 387)], [(102, 131)], [(16, 41), (83, 99), (139, 180), (187, 236), (275, 292), (297, 368)],
    [(114, 146)], [(0, 395)], [(174, 295)], [(118, 125), (126, 135), (373, 375)], [(241, 395)], [(146, 217)],
    [(155, 162), (163, 168)], [(372, 385)], [(131, 150)], [(0, 395)], [(230, 311)],
    [(98, 101), (101, 106), (108, 111), (153, 155), (156, 162), (163, 168), (376, 378)], [(117, 262), (352, 354)],
    [(72, 74), (73, 75), (86, 88), (87, 89), (93, 95), (94, 96), (107, 109), (108, 110), (127, 129), (128, 130),
     (129, 131), (142, 144), (143, 145), (156, 158), (157, 159), (170, 172), (171, 173), (183, 185), (184, 186),
     (185, 187), (197, 199), (198, 200), (199, 201), (204, 206), (205, 207), (206, 208), (233, 235), (234, 236),
     (247, 249), (248, 250), (261, 263), (262, 264), (275, 277), (276, 278), (282, 284), (283, 285), (303, 305),
     (304, 306), (305, 307), (309, 311), (310, 312), (311, 313), (324, 326), (325, 327)], [(105, 386)], [(30, 73)],
    [(3, 38), (102, 204), (283, 303)], [(7, 59), (174, 197), (257, 314), (342, 387)], [(0, 159)], [(166, 189)],
    [(60, 86)], [(59, 107), (189, 252), (360, 363)], [(48, 111), (210, 252)], [(62, 76), (282, 322)], [(88, 195)],
    [(93, 395)], [(260, 339)], [(37, 187)], [(39, 118), (237, 349)], [(160, 395)], [(61, 78), (148, 290), (325, 373)],
    [(51, 113), (208, 250), (333, 372)], [(25, 395)], [(248, 264)], [(104, 166)], [(0, 334)], [(26, 53)],
    [(3, 5), (8, 10), (10, 12), (15, 17), (17, 19), (23, 25), (24, 26), (26, 28), (29, 31), (31, 33), (33, 35),
     (54, 56), (59, 61), (61, 63), (64, 66), (73, 75), (75, 77), (76, 78), (79, 81), (80, 82), (83, 85), (90, 92),
     (91, 93), (94, 96), (95, 97), (98, 100), (100, 102), (101, 103), (119, 121), (211, 213), (247, 249), (253, 255),
     (254, 256), (255, 257), (257, 259), (260, 262), (267, 269), (269, 271), (275, 277), (276, 278), (283, 285),
     (285, 287), (288, 290), (289, 291), (290, 292), (295, 297), (296, 298), (297, 299), (299, 301), (304, 306),
     (314, 316), (316, 318), (318, 320), (322, 324), (325, 327), (327, 329), (330, 332), (331, 333), (332, 334),
     (337, 339), (339, 341), (344, 346), (346, 348), (358, 360), (360, 362), (362, 364), (367, 369), (372, 374),
     (374, 376), (379, 381), (381, 383), (386, 388), (387, 389), (388, 390), (390, 392), (393, 395), (394, 395)],
    [(99, 111)], [(274, 377)], [(65, 82), (196, 221), (314, 365)], [(141, 232)],
    [(2, 5), (9, 12), (16, 18), (17, 19), (23, 25), (24, 26), (30, 33), (37, 40), (44, 47), (51, 54), (58, 61),
     (65, 68), (72, 75), (79, 82), (86, 89), (93, 95), (94, 96), (100, 103), (107, 110), (109, 111), (114, 117),
     (121, 123), (122, 124), (128, 131), (135, 138), (142, 145), (149, 152), (163, 166), (170, 173), (177, 180),
     (184, 187), (206, 208), (213, 215), (219, 222), (226, 228), (227, 229), (233, 236), (240, 243), (247, 249),
     (248, 250), (254, 257), (261, 263), (262, 264), (268, 271), (275, 278), (282, 285), (289, 291), (290, 292),
     (296, 299), (303, 306), (310, 313), (317, 320), (319, 321), (324, 327), (331, 334), (338, 341), (345, 348),
     (352, 355), (358, 361), (360, 362), (366, 368), (367, 369), (373, 376), (380, 383), (387, 390), (394, 395)],
    [(101, 120)], [(165, 227)], [(233, 339)], [(277, 342)], [(304, 381)],
    [(30, 32), (51, 53), (52, 54), (58, 61), (65, 68), (79, 82), (81, 83), (86, 89), (93, 96), (100, 103), (107, 109),
     (114, 116), (115, 117), (120, 122), (121, 123), (122, 124), (135, 137), (136, 138), (142, 145), (149, 152),
     (151, 153), (153, 156), (156, 159), (158, 161), (160, 162), (161, 163), (162, 164), (163, 166), (165, 168),
     (168, 170), (169, 172), (171, 173), (172, 175), (174, 177), (177, 180), (180, 182), (184, 186), (188, 190),
     (190, 193), (192, 194), (193, 195), (194, 196), (204, 206), (205, 207), (206, 208), (212, 214), (213, 215),
     (219, 222), (226, 228), (231, 233), (233, 236), (241, 243), (244, 246), (254, 256), (260, 262), (261, 264),
     (265, 267), (268, 270), (275, 277), (289, 291), (290, 292), (296, 298), (303, 305), (304, 306), (324, 327),
     (374, 376)],
    [(0, 2), (2, 4), (3, 5), (7, 9), (10, 12), (14, 16), (17, 19), (21, 23), (24, 26), (28, 30), (31, 33), (35, 37),
     (38, 40), (42, 44), (44, 46), (45, 47), (49, 51), (52, 54), (56, 58), (58, 60), (59, 61), (63, 65), (66, 68),
     (70, 72), (73, 75), (77, 79), (80, 82), (84, 86), (87, 89), (91, 93), (94, 96), (98, 100), (101, 103), (105, 107),
     (108, 110), (109, 111), (112, 114), (115, 117), (118, 120), (122, 124), (129, 131), (135, 137), (136, 138),
     (140, 142), (143, 145), (147, 149), (150, 152), (154, 156), (157, 159), (161, 163), (164, 166), (168, 170),
     (170, 172), (171, 173), (175, 177), (177, 179), (178, 180), (182, 184), (185, 187), (187, 189), (192, 194),
     (196, 198), (199, 201), (203, 205), (206, 208), (213, 215), (217, 219), (220, 222), (227, 229), (231, 233),
     (234, 236), (238, 240), (241, 243), (245, 247), (248, 250), (252, 254), (255, 257), (259, 261), (262, 264),
     (266, 268), (269, 271), (273, 275), (276, 278), (280, 282), (282, 284), (283, 285), (287, 289), (290, 292),
     (294, 296), (297, 299), (301, 303), (303, 305), (304, 306), (308, 310), (311, 313), (315, 317), (317, 319),
     (318, 320), (322, 324), (325, 327), (329, 331), (332, 334), (336, 338), (339, 341), (343, 345), (345, 347),
     (346, 348), (350, 352), (353, 355), (360, 362), (363, 365), (364, 366), (367, 369), (371, 373), (374, 376),
     (381, 383), (388, 390), (392, 394)],
    [(24, 26), (31, 33), (37, 39), (45, 47), (51, 53), (52, 54), (59, 61), (65, 68), (72, 74), (73, 75), (79, 82),
     (81, 83), (86, 89), (93, 96), (100, 103), (102, 104), (107, 110), (109, 111), (114, 116), (115, 117), (116, 118),
     (119, 121), (121, 124), (128, 131), (135, 137), (136, 138), (143, 145), (149, 151), (163, 165), (170, 172),
     (178, 180), (179, 181), (184, 187), (192, 194), (206, 208), (212, 214), (213, 215), (220, 222), (226, 228),
     (227, 229), (233, 235), (234, 236), (240, 243), (254, 256), (255, 257), (261, 263), (262, 264), (268, 270),
     (269, 271), (275, 277), (276, 278), (289, 292), (291, 293), (296, 298), (297, 299), (303, 306), (310, 312),
     (311, 313), (318, 320), (324, 326), (325, 327), (331, 333), (332, 334), (345, 348), (374, 376), (381, 383)],
    [(1, 3), (4, 6), (9, 11), (16, 19), (23, 26), (30, 33), (38, 40), (42, 44), (44, 47), (46, 49), (48, 51), (50, 52),
     (51, 54), (53, 55), (58, 61), (61, 63), (65, 68), (67, 70), (72, 75), (76, 78), (79, 82), (86, 89), (93, 96),
     (100, 103), (106, 108), (108, 110), (114, 117), (119, 121), (121, 124), (126, 128), (128, 131), (136, 138),
     (142, 145), (150, 152), (163, 165), (164, 166), (170, 173), (184, 187), (192, 194), (205, 207), (212, 214),
     (220, 222), (226, 229), (233, 236), (240, 243), (247, 249), (254, 257), (261, 264), (268, 271), (275, 278),
     (289, 292), (296, 299), (303, 306), (310, 313), (317, 320), (319, 321), (324, 327), (331, 334), (338, 341),
     (346, 348), (352, 355), (359, 362), (366, 369), (374, 376), (380, 383), (387, 390)],
    [(28, 31), (30, 32), (31, 33), (32, 34), (332, 335), (334, 336), (336, 338), (358, 361), (361, 364), (364, 366),
     (366, 368), (368, 370), (371, 373)], [(65, 73), (187, 232), (324, 381)],
    [(9, 11), (16, 18), (23, 25), (30, 33), (45, 47), (51, 54), (57, 59), (58, 60), (59, 61), (66, 68), (72, 74),
     (73, 75), (77, 80), (79, 82), (86, 89), (93, 95), (94, 96), (99, 101), (100, 103), (107, 110), (109, 111),
     (114, 117), (119, 121), (121, 123), (122, 124), (128, 131), (135, 138), (142, 144), (143, 145), (149, 151),
     (150, 152), (154, 156), (203, 206), (205, 208), (211, 213), (212, 215), (217, 220), (219, 222), (226, 228),
     (227, 229), (231, 234), (233, 236), (238, 241), (240, 243), (254, 256), (255, 257), (261, 264), (268, 270),
     (275, 277), (276, 278), (289, 292), (295, 297), (296, 298), (297, 299), (303, 305), (304, 306), (310, 313),
     (324, 327), (331, 334), (338, 340), (339, 341), (345, 347), (352, 354), (373, 375), (380, 382), (387, 389)],
    [(103, 163)], [(0, 319)], [(0, 333)], [(122, 222)], [(203, 323)], [(20, 50), (191, 235), (277, 328)], [(0, 379)],
    [(10, 12), (17, 19), (30, 33), (37, 39), (45, 47), (51, 53), (52, 54), (53, 55), (58, 60), (59, 61), (65, 67),
     (66, 68), (67, 69), (71, 73), (72, 75), (79, 81), (80, 82), (81, 83), (86, 89), (93, 96), (100, 103), (107, 109),
     (108, 111), (114, 116), (115, 117), (119, 121), (121, 124), (128, 131), (143, 145), (149, 151), (163, 166),
     (170, 173), (177, 180), (184, 186), (206, 208), (212, 214), (213, 215), (219, 221), (220, 222), (226, 228),
     (227, 229), (233, 236), (240, 242), (241, 243), (248, 250), (254, 256), (255, 257), (261, 264), (268, 271),
     (275, 278), (282, 284), (284, 286), (289, 292), (295, 297), (296, 299), (303, 306), (310, 313), (319, 321),
     (324, 327), (331, 334), (338, 340), (373, 376), (380, 383), (387, 389), (388, 390), (394, 395)],
    [(1, 3), (6, 8), (13, 15), (20, 22), (27, 30), (34, 36), (36, 38), (40, 43), (47, 50), (49, 52), (55, 57), (62, 64),
     (69, 71), (76, 78), (83, 85), (90, 92), (96, 98), (98, 100), (104, 107), (109, 111), (111, 113), (117, 120),
     (124, 128), (132, 134), (139, 141), (146, 148), (152, 154), (160, 162), (167, 169), (174, 176), (180, 183),
     (188, 190), (194, 198), (201, 204), (203, 206), (209, 212), (216, 218), (223, 225), (230, 232), (238, 240),
     (244, 246), (250, 253), (252, 255), (256, 260), (265, 267), (272, 274), (279, 281), (286, 288), (292, 296),
     (299, 302), (307, 309), (313, 316), (315, 318), (321, 323), (326, 331), (334, 337), (342, 344), (349, 351),
     (355, 357), (362, 364), (369, 371), (371, 373), (377, 379), (384, 386), (389, 392), (391, 394)],
    [(4, 6), (6, 9), (10, 12), (15, 18), (19, 22), (23, 25), (28, 30), (32, 35), (34, 37), (36, 38), (40, 43), (42, 45),
     (47, 50), (51, 53), (62, 65), (76, 78), (82, 85), (84, 86), (86, 88), (111, 113), (125, 127), (131, 134),
     (133, 135), (144, 146), (159, 162), (162, 164), (168, 170), (194, 196), (195, 198), (202, 205), (204, 206),
     (209, 212), (218, 220), (237, 239), (243, 245), (251, 253), (257, 260), (260, 262), (265, 267), (266, 269),
     (279, 281), (293, 295), (294, 297), (297, 299), (301, 304), (307, 309), (308, 311), (310, 312), (323, 325),
     (334, 337), (336, 339), (339, 341), (341, 344), (344, 346), (354, 356), (386, 389)], [(123, 287)], [(285, 355)],
    [(2, 5), (10, 12), (15, 18), (17, 19), (23, 26), (30, 33), (43, 45), (44, 47), (47, 49), (52, 54), (59, 61),
     (65, 67), (67, 69), (70, 72), (72, 75), (75, 78), (79, 82), (83, 85), (85, 88), (91, 93), (94, 96), (100, 103),
     (107, 109), (114, 116), (115, 117), (119, 121), (121, 123), (128, 130), (129, 131), (130, 132), (135, 138),
     (140, 142), (142, 144), (143, 145), (233, 235), (255, 257), (262, 264), (268, 271), (276, 278), (282, 284),
     (283, 285), (290, 292), (296, 299), (304, 306), (310, 313), (316, 318), (317, 320), (319, 321), (323, 326),
     (325, 327), (327, 329), (331, 333), (332, 335), (339, 341), (345, 348), (352, 354), (353, 355), (359, 361),
     (361, 364), (366, 369), (369, 371), (373, 375), (374, 376), (380, 383), (388, 390), (393, 395)], [(249, 332)],
    [(238, 371)],
    [(0, 1), (2, 5), (9, 12), (16, 18), (24, 26), (30, 33), (37, 40), (40, 43), (42, 45), (44, 47), (48, 51), (51, 53),
     (54, 56), (57, 59), (59, 61), (65, 67), (66, 69), (72, 75), (79, 82), (86, 89), (93, 96), (100, 103), (109, 111),
     (114, 117), (121, 123), (128, 130), (129, 131), (135, 138), (142, 145), (149, 152), (163, 166), (170, 173),
     (185, 187), (191, 194), (199, 201), (213, 215), (219, 222), (226, 228), (227, 229), (233, 236), (241, 243),
     (247, 250), (255, 257), (261, 264), (275, 278), (290, 292), (296, 299), (303, 306), (310, 313), (317, 320),
     (324, 327), (331, 333), (332, 334), (338, 341), (345, 348), (352, 355), (360, 362), (367, 369), (369, 371),
     (372, 374), (373, 375), (374, 376), (380, 382), (387, 389), (388, 390), (394, 395)], [(13, 395)],
    [(58, 168), (277, 303), (388, 391)]
]

GREEDY_EVENT_KEYWORDS = [
    ['pevný', 'květen', 'červen', 'předpokládat', 'model', 'vůz', 'automobil', 'značka', 'technologie', 'výroba',
     'verze', 'množství', 'klasický', 'technický', 'vývoj', 'střední'],
    ['záležet', 'leden', 'zimní', 'zima'],
    ['zemřít', 'září', 'vydat', 'Praha', 'město', 'komentář', 'nový', 'uvést', 'informace', 'rok', 'případ'],
    ['patřit', 'svět', 'místo', 'ještě'],
    ['novinka', 'společnost', 'zdroj', 'zpráva'],
    ['Silvestr', 'prosinec', 'Vánoce', 'dárek'],
    ['irácký', 'srpen', 'červenec', 'letní', 'prázdniny'],
    ['rozvod', 'duben', 'březen', 'jarní'],
    ['lidovec', 'říjen', 'zastupitelstvo', 'koalice', 'hnutí', 'volba', 'hlas'],
    ['smlouva', 'firma', 'trh', 'cena', 'hodnota', 'rámec', 'koruna', 'milión', 'služba', 'systém', 'projekt'],
    ['změna', 'číslo', 'zákon', 'veřejný', 'státní', 'úřad', 'republika', 'vláda', 'sociální', 'finanční',
     'ministerstvo', 'ředitel'],
    ['pololetí', 'listopad', 'Zeman', 'Miloš'],
    ['Seat', 'únor', 'loňský', 'vloni'],
    ['vidět', 'dát', 'tam', 'dobře', 'vědět'],
    ['tiskový', 'miliarda', 'růst', 'obchodní', 'zákazník'],
    ['zářijový', 'vánoční', 'svátek'],
    ['řidič', 'kolo', 'hráč', 'sice', 'stačit', 'hrát', 'tým', 'poslední', 'dokázat', 'přijít', 'hned', 'nakonec'],
    ['vyrábět', 'využívat', 'typ', 'program', 'různý', 'zajímavý', 'objevit', 'přímo', 'základní', 'oblast', 'týkat'],
    ['Američan', 'vyhrát', 'vítězství', 'zápas', 'minuta', 'utkání', 'soupeř'],
    ['Synot', 'aktualizace', 'příští', 'letos', 'web', 'ČTK', 'dítě', 'opět', 'hodina', 'autor', 'akce', 'strana',
     'týden', 'mladý'],
    ['Mercedes', 'sobota', 'neděle', 'pátek', 'sezona', 'šance', 'divák', 'domácí', 'doma', 'závěr', 'moc', 'vést',
     'hlava', 'skončit', 'těžký'],
    ['cukroví', 'štědrý', 'sníh', 'mráz'],
    ['údajně', 'Evropa', 'evropský', 'světový', 'získat', 'výsledek', 'podařit', 'výkon', 'tady'],
    ['klesnout', 'pokles', 'vzrůst', 'banka', 'ekonomika', 'dolar', 'euro', 'procento', 'výše', 'klient'],
    ['náměstí', 'prezident', 'Václav', 'událost', 'pražský', 'Martin', 'vedení', 'organizace', 'škola'],
    ['využívající', 'pohon', 'automobilový', 'automobilka'],
    ['přesilovka', 'gól', 'branka'],
    ['prázdninový', 'festival', 'dovolená', 'léto', 'voda', 'návštěvník', 'ročník'],
    ['pracovní', 'spolupráce', 'výrazný', 'přinášet', 'přicházet', 'kategorie', 'funkce', 'nízký', 'provoz', 'velikost',
     'šéf', 'nabídka'],
    ['sen', 'myšlenka', 'hodit', 'složitý', 'drahý'],
    ['odstavit', 'perfektně', 'úsporný', 'modernizace', 'zachování', 'chod', 'rozumný', 'splňovat', 'komplexní',
     'rozšířený', 'obdobný', 'infrastruktura', 'výbava'],
    ['potrubí', 'atmosférický', 'agregát', 'Peugeot', 'hybridní', 'Audi', 'koncern', 'modelový', 'exkluzivní'],
    ['srpnový', 'sankce', 'Rusko', 'ruský'],
    ['mimořádně', 'lákat', 'stálý', 'absolutní', 'extrémní', 'dno', 'otevření', 'zadní', 'střed', 'brána', 'jednička',
     'zúčastnit', 'přední'],
    ['puk', 'led', 'hokejista', 'hokejový'],
    ['výrobce', 'jednotka', 'sportovní', 'prohlásit', 'hodlat', 'fáze', 'postup'],
    ['siga', 'Tweet', 'top', 'podnik', 'fórum', 'reklama', 'zpravodajství', 'klíčový', 'Brno', 'hranice', 'bod', 'hra',
     'fotogalerie', 'chyba', 'boj'],
    ['půle', 'poločas', 'míč', 'trefit', 'výhra', 'prohrát', 'duel', 'porazit', 'souboj'],
    ['studie', 'test', 'třída', 'energie', 'délka', 'motor', 'elektrický'],
    ['příčka', 'tabulka', 'host', 'trenér', 'sestava'],
    ['akciový', 'prodej', 'řízení'],
    ['mercedes', 'sériový', 'přeplňovaný', 'označení', 'ekologický', 'prezentovat', 'parametr', 'standardní',
     'spotřeba', 'poněkud', 'typický', 'strategie', 'moment', 'podání', 'povinný', 'dvojice', 'stopa', 'snižovat'],
    ['policie', 'soud', 'jednání', 'návrh', 'předseda'],
    ['sazba', 'akcie', 'investor'],
    ['zisk', 'středa', 'čtvrtek', 'včera'],
    ['Golf', 'karosérie', 'komfort', 'akumulátor'],
    ['titul', 'série', 'liga', 'hřiště', 'fotbalista', 'fotbalový'],
    ['přelomový', 'euforie', 'bavorský', 'nastartovat', 'oceňovat', 'revoluční', 'inovativní', 'vývojový', 'vybudování',
     'laboratoř', 'průběžně', 'vrcholný', 'reprezentovat', 'World', 'odvážný', 'ambice', 'vylepšit'],
    ['Jágr', 'olympijský', 'olympiáda'],
    ['sekunda', 'závod', 'start'],
    ['koncernový', 'SUV', 'autosalón', 'Volkswagen', 'prototyp', 'plast', 'pneumatika', 'náprava', 'ručně', 'repertoár',
     'velkolepý', 'říše', 'terénní', 'práh', 'stěhovat', 'zvedat', 'připojovat', 'salón'],
    ['triumf', 'finále', 'mistrovství', 'šampionát', 'turnaj'],
    ['sametový', 'revoluce', 'Havel', 'výročí'],
    ['úterní', 'úterý', 'pondělí'],
    ['Nizozemsko', 'Brazílie', 'fotbal', 'Německo', 'Francie', 'území', 'východ', 'armáda', 'vojenský', 'voják'],
    ['součinitel', 'Toyota', 'prodloužený'],
    ['KDU-ČSL', 'ČSSD', 'ODS'],
    ['koleda', 'Ukrajina', 'ukrajinský'],
    ['derby', 'Sparta', 'ligový', 'brankář', 'gólman', 'střela', 'obránce', 'vyrovnat'],
    ['MQB', 'užitkový', 'osudový', 'charakterizovat', 'terén', 'luxus', 'kupé', 'prostorný', 'Renault', 'nanometr',
     'točivý'],
    ['Ježíšek', 'islámský', 'islamista', 'Sýrie', 'Irák'],
    ['Vancouver', 'lyže', 'medaile', 'pohár', 'radovat', 'útočník', 'zvítězit', 'porážka', 'náskok', 'slavit', 'jaro',
     'Plzeň', 'účast', 'odpovědět'],
    ['extraliga', 'play', 'rozhodčí', 'skóre', 'trefa'],
    ['doněcký', 'povstalec', 'Kyjev', 'Moskva', 'konflikt', 'západ'],
    ['Luhansk', 'Krym', 'Putin', 'Rus'],
    ['Slavia', 'sobotní', 'nedělní', 'páteční'],
    ['volit', 'volební', 'volič'],
    ['dveřový', 'Citroen', 'zvýraznění', 'vyhledat', 'drát', 'provedený', 'palivo', 'zastavení', 'mezitím', 'norma',
     'získaný', 'předpis', 'úplný', 'rozvíjet', 'posunout', 'pole', 'vzduch'],
    ['Doněck', 'letadlo', 'letecký'],
    ['automobilismus', 'benzínový', 'redukce', 'doplnění', 'spojka', 'dotyk', 'prodlužovat'],
    ['kliknout', 'vložit', 'student', 'kampaň', 'post', 'projev'],
    ['poselství', 'zveřejněný', 'plyn', 'objem', 'závislost', 'zavést', 'Brusel', 'francouzský', 'centimetr'],
    ['prius', 'hybrid', 'dohánět', 'čerpací', 'zpoždění'],
    ['Viktor', 'parlament', 'kandidát', 'zelený'],
    ['Ces', 'lednový', 'ropa', 'útok', 'oběť'],
    ['sestřelit', 'raketa', 'Izrael', 'izraelský'],
    ['cyklista', 'komunální', 'podzimní', 'senát', 'kandidátka', 'mandát'],
    ['únorový', 'semifinále', 'japonský', 'nárůst', 'oslava'],
    ['zvětšený', 'květnový', 'referendum', 'NATO', 'Paříž', 'terorista', 'teroristický', 'muslim', 'islám'],
    ['boeing', 'předplatné', 'extra', 'Babiš', 'opozice', 'demonstrace'],
    ['Minsk', 'Peking', 'maďarský', 'shlédnout'],
    ['posunutý', 'Janukovyč', 'demonstrant', 'tyč', 'humanitární', 'Litvínov', 'palestinský']
]

GREEDY_EVENT_BURSTS = [
    [(127, 172)], [(0, 395)], [(241, 273)],
    [(0, 1), (2, 4), (3, 5), (10, 12), (24, 26), (30, 33), (37, 40), (44, 47), (51, 54), (65, 68), (87, 89), (94, 96),
     (107, 109), (108, 110), (114, 117), (119, 121), (122, 124), (126, 128), (127, 129), (128, 131), (130, 133),
     (132, 135), (135, 138), (140, 143), (142, 144), (143, 145), (144, 146), (148, 150), (149, 151), (150, 152),
     (151, 153), (152, 154), (156, 159), (161, 163), (163, 165), (164, 166), (165, 167), (170, 173), (178, 180),
     (184, 187), (191, 194), (198, 201), (206, 208), (227, 229), (248, 250), (269, 271), (283, 285), (297, 299),
     (310, 312), (311, 313), (317, 320), (346, 348), (352, 355), (356, 358), (357, 359), (359, 361), (363, 366),
     (366, 368), (373, 375), (374, 376), (379, 381), (380, 383), (383, 386), (385, 388), (387, 390), (394, 395)],
    [(0, 2), (4, 6), (12, 14), (20, 22), (25, 27), (27, 29), (34, 37), (41, 44), (48, 50), (50, 52), (54, 57),
     (62, 64), (69, 71), (75, 78), (77, 80), (82, 85), (88, 91), (90, 93), (92, 94), (95, 98), (97, 100), (103, 106),
     (110, 113), (113, 119), (118, 122), (123, 126), (131, 133), (133, 136), (138, 140), (146, 148), (151, 154),
     (159, 162), (162, 164), (166, 168), (173, 176), (176, 178), (181, 183), (187, 189), (195, 198), (203, 205),
     (208, 213), (215, 217), (223, 225), (229, 232), (237, 240), (244, 246), (249, 252), (251, 254), (257, 260),
     (265, 268), (271, 275), (279, 282), (285, 289), (292, 296), (293, 299), (301, 303), (306, 310), (313, 315),
     (315, 317), (316, 318), (321, 323), (327, 330), (335, 337), (341, 343), (343, 346), (348, 350)], [(163, 395)],
    [(186, 237)], [(64, 111)], [(273, 304)],
    [(4, 6), (13, 15), (19, 24), (26, 30), (33, 37), (42, 49), (55, 57), (57, 59), (62, 64), (69, 72), (75, 79),
     (83, 85), (88, 90), (90, 93), (96, 100), (103, 106), (110, 113), (113, 119), (119, 122), (124, 126), (127, 131),
     (131, 135), (138, 140), (140, 143), (146, 149), (151, 154), (154, 156), (159, 161), (161, 164), (165, 168),
     (167, 170), (175, 177), (181, 184), (187, 192), (195, 197), (202, 204), (208, 212), (215, 218), (223, 225),
     (229, 232), (237, 239), (244, 246), (249, 253), (256, 260), (263, 265), (265, 268), (271, 274), (279, 281),
     (286, 288), (292, 296), (301, 304), (306, 310), (313, 316), (321, 324), (326, 329), (328, 331), (336, 338),
     (340, 342), (342, 345), (349, 352), (347, 356), (369, 371), (375, 378), (377, 380), (384, 387), (391, 394)],
    [(0, 2), (4, 6), (6, 10), (12, 15), (19, 23), (25, 28), (27, 30), (33, 36), (35, 38), (39, 42), (42, 45), (47, 50),
     (50, 52), (55, 57), (57, 59), (63, 65), (75, 78), (83, 86), (88, 91), (91, 93), (96, 98), (97, 100), (103, 105),
     (112, 114), (116, 119), (123, 126), (132, 134), (140, 142), (146, 148), (167, 169), (181, 184), (195, 197),
     (203, 205), (209, 211), (252, 254), (265, 269), (271, 273), (273, 275), (278, 280), (280, 282), (286, 289),
     (291, 294), (293, 295), (294, 297), (298, 300), (301, 304), (306, 309), (313, 316), (316, 318), (321, 324),
     (327, 330), (329, 332), (334, 337), (336, 339), (341, 345), (348, 350), (350, 353), (354, 357), (361, 363),
     (369, 372), (375, 377), (377, 380), (383, 386), (385, 388), (389, 392), (392, 394)], [(304, 332)], [(0, 302)],
    [(0, 1), (2, 5), (9, 12), (16, 19), (23, 26), (31, 33), (37, 40), (45, 49), (51, 53), (58, 61), (65, 68), (72, 75),
     (79, 82), (86, 89), (93, 96), (100, 103), (108, 110), (114, 117), (120, 124), (128, 131), (135, 138), (142, 144),
     (143, 145), (149, 152), (156, 159), (163, 166), (170, 173), (178, 180), (184, 187), (191, 193), (206, 208),
     (212, 215), (219, 222), (226, 229), (233, 236), (240, 243), (247, 250), (255, 257), (262, 264), (268, 271),
     (275, 278), (282, 285), (289, 292), (297, 299), (303, 306), (310, 313), (317, 319), (318, 321), (324, 327),
     (331, 334), (338, 341), (345, 348), (352, 355), (356, 358), (358, 360), (359, 362), (363, 366), (366, 369),
     (373, 376), (375, 377), (380, 382), (382, 384), (385, 387), (388, 390), (391, 393), (393, 395)],
    [(0, 2), (5, 10), (11, 14), (14, 16), (19, 21), (21, 24), (27, 30), (33, 37), (40, 45), (46, 50), (54, 58),
     (62, 65), (68, 72), (76, 79), (83, 86), (89, 93), (95, 98), (97, 100), (104, 106), (111, 114), (117, 119),
     (124, 126), (127, 129), (131, 135), (138, 140), (147, 149), (152, 156), (158, 161), (161, 168), (167, 171),
     (173, 180), (181, 184), (187, 194), (195, 197), (201, 205), (208, 212), (214, 216), (216, 219), (222, 226),
     (229, 232), (235, 239), (242, 246), (250, 253), (256, 258), (259, 261), (264, 267), (271, 275), (278, 281),
     (286, 288), (292, 296), (298, 304), (307, 310), (313, 317), (321, 323), (327, 329), (329, 332), (335, 338),
     (341, 345), (348, 352), (354, 357), (362, 366), (369, 373), (372, 374), (377, 381), (383, 387), (391, 394)],
    [(271, 395)],
    [(3, 5), (9, 12), (16, 19), (23, 26), (30, 33), (37, 40), (44, 47), (51, 54), (58, 61), (65, 68), (72, 75),
     (79, 82), (86, 89), (93, 96), (100, 103), (107, 109), (108, 110), (109, 111), (114, 117), (116, 118), (119, 121),
     (121, 123), (122, 124), (127, 129), (128, 131), (130, 133), (132, 135), (135, 137), (136, 138), (142, 144),
     (143, 145), (144, 146), (148, 150), (149, 152), (151, 154), (163, 166), (177, 179), (205, 207), (219, 222),
     (226, 229), (233, 236), (240, 243), (254, 257), (261, 264), (268, 270), (269, 271), (275, 278), (282, 285),
     (288, 290), (289, 292), (296, 299), (303, 306), (310, 313), (318, 320), (324, 326), (325, 327), (331, 334),
     (338, 341), (352, 355), (358, 360), (360, 362), (367, 369), (373, 376), (380, 383), (387, 390), (394, 395)],
    [(116, 156)],
    [(2, 4), (3, 5), (9, 12), (16, 19), (23, 25), (24, 26), (30, 33), (37, 40), (44, 47), (51, 54), (58, 60), (59, 61),
     (65, 68), (72, 75), (79, 82), (86, 88), (87, 89), (93, 96), (100, 102), (101, 103), (107, 110), (109, 111),
     (114, 117), (121, 124), (128, 131), (135, 138), (142, 144), (143, 145), (144, 146), (149, 152), (163, 166),
     (170, 173), (177, 179), (184, 187), (192, 194), (219, 222), (226, 229), (233, 236), (240, 243), (248, 250),
     (254, 257), (261, 264), (268, 271), (275, 278), (282, 285), (289, 292), (295, 297), (296, 299), (303, 306),
     (310, 313), (317, 320), (319, 321), (324, 327), (331, 334), (338, 341), (345, 348), (352, 355), (358, 360),
     (360, 362), (366, 369), (373, 376), (380, 383), (387, 389), (388, 390), (389, 391), (394, 395)], [(264, 359)],
    [(2, 5), (9, 12), (16, 19), (23, 26), (30, 33), (37, 40), (44, 47), (51, 54), (58, 61), (65, 68), (72, 75),
     (79, 82), (81, 83), (86, 89), (93, 96), (100, 102), (101, 103), (102, 104), (106, 109), (108, 111), (113, 115),
     (114, 117), (116, 118), (120, 124), (128, 131), (135, 138), (141, 144), (143, 145), (149, 152), (156, 159),
     (163, 166), (170, 173), (177, 180), (184, 187), (191, 194), (198, 201), (205, 208), (212, 215), (219, 222),
     (226, 229), (233, 236), (240, 243), (247, 250), (254, 257), (261, 264), (268, 271), (275, 278), (281, 283),
     (282, 285), (289, 292), (296, 299), (303, 306), (310, 313), (316, 318), (318, 320), (324, 327), (331, 333),
     (338, 341), (345, 348), (352, 355), (359, 362), (366, 369), (373, 376), (380, 383), (387, 390), (394, 395)],
    [(39, 395)], [(113, 154), (284, 395)],
    [(1, 3), (5, 9), (12, 16), (19, 23), (25, 28), (28, 30), (33, 36), (40, 44), (48, 50), (53, 55), (55, 58),
     (62, 65), (68, 72), (75, 79), (84, 86), (89, 91), (96, 101), (103, 106), (111, 114), (117, 121), (124, 127),
     (130, 132), (132, 135), (138, 141), (147, 149), (154, 156), (159, 164), (166, 168), (180, 184), (186, 189),
     (195, 198), (202, 204), (207, 209), (210, 212), (214, 217), (216, 219), (224, 226), (229, 233), (238, 240),
     (249, 254), (258, 261), (265, 267), (272, 274), (278, 281), (284, 286), (286, 289), (293, 295), (295, 297),
     (299, 304), (306, 308), (308, 311), (313, 317), (321, 324), (326, 331), (334, 338), (340, 343), (343, 346),
     (348, 352), (354, 357), (361, 366), (369, 373), (375, 377), (377, 380), (382, 384), (384, 387), (390, 394)],
    [(145, 395)], [(118, 159)],
    [(2, 4), (3, 5), (9, 12), (16, 18), (17, 19), (23, 26), (30, 33), (37, 39), (44, 47), (51, 54), (58, 60), (59, 61),
     (65, 67), (66, 68), (71, 74), (73, 75), (79, 82), (83, 85), (86, 89), (93, 95), (94, 96), (100, 103), (107, 110),
     (109, 111), (114, 117), (119, 121), (121, 124), (128, 131), (135, 138), (142, 144), (143, 145), (163, 165),
     (164, 166), (170, 173), (220, 222), (226, 228), (233, 236), (240, 243), (254, 256), (255, 257), (261, 264),
     (268, 270), (269, 271), (275, 278), (282, 285), (289, 292), (296, 299), (299, 301), (303, 306), (310, 313),
     (317, 319), (318, 320), (319, 321), (324, 327), (331, 334), (338, 340), (339, 341), (345, 348), (352, 355),
     (358, 361), (360, 362), (366, 369), (373, 376), (380, 383), (387, 390), (394, 395)], [(161, 219)],
    [(113, 133), (141, 164)], [(104, 129), (125, 152)], [(116, 134)], [(110, 155)], [(56, 89), (208, 251), (328, 362)],
    [(128, 136), (140, 147), (149, 154)], [(4, 320)], [(110, 145)], [(36, 97), (278, 335)],
    [(2, 5), (9, 12), (16, 19), (23, 26), (30, 33), (37, 40), (44, 47), (51, 54), (58, 60), (59, 61), (65, 68),
     (67, 69), (72, 75), (79, 82), (86, 89), (93, 96), (100, 103), (107, 109), (108, 110), (109, 111), (114, 117),
     (121, 123), (122, 124), (128, 131), (135, 138), (142, 144), (143, 145), (149, 151), (163, 166), (170, 173),
     (177, 179), (184, 186), (213, 215), (220, 222), (226, 228), (227, 229), (233, 236), (240, 243), (254, 257),
     (261, 263), (262, 264), (268, 270), (269, 271), (275, 278), (282, 285), (289, 292), (296, 299), (303, 306),
     (310, 312), (311, 313), (317, 319), (318, 320), (319, 321), (324, 327), (331, 334), (338, 341), (345, 348),
     (352, 355), (358, 360), (360, 362), (367, 369), (373, 376), (380, 382), (381, 383), (387, 390), (394, 395)],
    [(117, 160), (319, 322)],
    [(2, 4), (3, 5), (9, 11), (10, 12), (16, 18), (17, 19), (23, 26), (30, 32), (31, 33), (37, 40), (44, 47), (51, 54),
     (58, 60), (59, 61), (65, 68), (67, 69), (72, 74), (73, 75), (79, 82), (86, 89), (93, 95), (94, 96), (100, 103),
     (107, 110), (109, 111), (114, 117), (121, 124), (128, 131), (135, 138), (143, 145), (170, 172), (226, 228),
     (233, 236), (241, 243), (254, 257), (261, 263), (262, 264), (268, 270), (269, 271), (275, 278), (283, 285),
     (289, 291), (290, 292), (295, 297), (296, 299), (303, 306), (310, 312), (311, 313), (317, 320), (324, 327),
     (330, 332), (331, 334), (338, 340), (339, 341), (345, 347), (346, 348), (352, 354), (353, 355), (358, 360),
     (360, 362), (366, 369), (373, 375), (374, 376), (380, 383), (387, 390), (394, 395)],
    [(8, 10), (11, 14), (18, 21), (34, 36), (36, 38), (41, 44), (55, 57), (63, 65), (69, 71), (75, 77), (82, 84),
     (84, 86), (89, 92), (91, 94), (97, 100), (103, 105), (104, 107), (110, 112), (111, 114), (113, 115), (117, 119),
     (125, 127), (127, 129), (128, 131), (130, 133), (132, 134), (133, 135), (139, 141), (140, 143), (142, 144),
     (143, 145), (144, 147), (146, 148), (149, 151), (151, 153), (152, 154), (153, 155), (158, 160), (161, 164),
     (165, 168), (167, 169), (181, 184), (202, 205), (266, 268), (271, 273), (277, 279), (286, 288), (292, 294),
     (295, 297), (298, 300), (306, 308), (307, 310), (314, 317), (321, 323), (322, 325), (326, 329), (337, 339),
     (340, 342), (343, 346), (349, 351), (351, 353), (370, 373), (372, 374), (376, 378), (379, 381), (383, 386)],
    [(117, 158)],
    [(5, 8), (13, 15), (19, 22), (22, 24), (27, 29), (34, 37), (40, 43), (42, 45), (47, 50), (49, 52), (53, 55),
     (54, 57), (56, 59), (60, 64), (75, 78), (78, 80), (82, 85), (84, 87), (89, 93), (96, 100), (103, 105), (117, 119),
     (125, 127), (131, 134), (139, 141), (146, 148), (154, 156), (165, 168), (168, 171), (172, 174), (174, 176),
     (180, 183), (195, 197), (202, 205), (204, 206), (207, 210), (209, 212), (230, 232), (243, 246), (252, 254),
     (257, 260), (263, 266), (266, 268), (271, 273), (279, 281), (285, 288), (288, 290), (292, 294), (293, 296),
     (295, 297), (301, 304), (307, 311), (312, 316), (315, 318), (321, 323), (323, 325), (327, 330), (336, 339),
     (342, 346), (347, 350), (370, 372), (377, 379), (384, 386), (386, 389), (389, 392), (391, 394)],
    [(0, 3), (5, 10), (11, 14), (14, 17), (20, 22), (25, 28), (27, 30), (33, 37), (41, 44), (48, 51), (55, 58),
     (61, 65), (69, 72), (75, 78), (82, 85), (84, 87), (89, 91), (91, 94), (96, 100), (104, 106), (111, 114),
     (117, 119), (121, 127), (131, 135), (138, 142), (146, 149), (152, 156), (158, 161), (162, 164), (166, 170),
     (173, 177), (179, 182), (182, 184), (187, 191), (195, 198), (202, 204), (210, 212), (216, 220), (222, 224),
     (225, 227), (230, 233), (236, 240), (244, 246), (251, 253), (257, 260), (259, 262), (264, 268), (272, 275),
     (279, 281), (286, 289), (293, 296), (298, 301), (301, 303), (307, 311), (314, 316), (321, 324), (327, 329),
     (333, 335), (341, 343), (343, 346), (348, 351), (350, 353), (361, 366), (378, 381), (384, 387), (392, 394)],
    [(0, 2), (7, 9), (13, 15), (14, 17), (21, 24), (28, 30), (35, 38), (42, 45), (49, 51), (55, 57), (56, 59),
     (62, 64), (63, 66), (70, 73), (77, 80), (84, 86), (91, 93), (98, 100), (105, 107), (111, 113), (112, 115),
     (119, 122), (126, 128), (133, 135), (140, 143), (147, 150), (154, 157), (161, 163), (168, 171), (175, 177),
     (188, 190), (189, 192), (196, 199), (198, 200), (203, 205), (210, 212), (217, 220), (224, 227), (231, 234),
     (238, 241), (243, 245), (245, 248), (252, 255), (259, 261), (266, 269), (273, 275), (280, 283), (287, 289),
     (293, 296), (295, 297), (301, 304), (308, 311), (314, 316), (315, 318), (320, 322), (322, 325), (329, 331),
     (335, 338), (337, 339), (343, 346), (350, 353), (371, 374), (378, 380), (384, 388), (391, 393), (392, 395)],
    [(143, 165)],
    [(17, 19), (23, 25), (51, 53), (52, 54), (58, 60), (59, 61), (65, 68), (67, 69), (72, 74), (73, 75), (79, 82),
     (83, 85), (86, 88), (87, 89), (93, 95), (94, 96), (99, 101), (100, 102), (101, 103), (107, 109), (108, 110),
     (109, 111), (110, 112), (113, 115), (114, 117), (116, 118), (117, 119), (118, 120), (119, 121), (120, 122),
     (121, 123), (122, 124), (123, 125), (128, 130), (129, 131), (135, 137), (136, 138), (142, 145), (148, 150),
     (149, 151), (150, 152), (151, 154), (205, 207), (212, 215), (219, 221), (220, 222), (226, 228), (233, 236),
     (240, 243), (254, 256), (261, 263), (262, 264), (268, 270), (275, 277), (276, 278), (289, 291), (290, 292),
     (296, 298), (297, 299), (303, 306), (310, 313), (324, 326), (325, 327), (331, 334), (338, 340), (352, 354)],
    [(116, 134)], [(21, 55)],
    [(2, 5), (9, 12), (17, 19), (31, 33), (37, 40), (40, 42), (43, 45), (44, 47), (47, 50), (51, 53), (65, 68),
     (72, 75), (93, 95), (101, 103), (108, 110), (113, 115), (114, 116), (115, 117), (116, 119), (118, 120),
     (119, 121), (120, 123), (122, 124), (123, 125), (127, 129), (128, 130), (129, 131), (130, 133), (132, 135),
     (134, 136), (135, 138), (143, 145), (149, 151), (150, 152), (156, 158), (157, 159), (158, 160), (161, 164),
     (163, 165), (164, 166), (165, 168), (170, 172), (171, 173), (177, 180), (185, 187), (192, 194), (199, 201),
     (206, 208), (220, 222), (227, 229), (233, 235), (234, 236), (247, 250), (255, 257), (262, 264), (269, 271),
     (276, 278), (283, 285), (325, 327), (338, 340), (339, 341), (352, 355), (373, 376), (381, 383), (387, 390),
     (394, 395)], [(131, 153)], [(104, 228)], [(303, 335)],
    [(5, 8), (12, 15), (19, 21), (26, 29), (33, 36), (40, 42), (46, 48), (48, 50), (54, 56), (61, 63), (68, 71),
     (74, 76), (75, 78), (82, 85), (89, 91), (95, 97), (96, 99), (102, 104), (103, 106), (109, 112), (111, 113),
     (117, 120), (124, 127), (131, 133), (138, 140), (145, 148), (152, 155), (159, 162), (166, 169), (173, 176),
     (180, 183), (187, 190), (194, 196), (200, 202), (201, 204), (208, 211), (215, 218), (222, 225), (229, 232),
     (235, 237), (236, 239), (242, 245), (247, 249), (250, 252), (257, 259), (264, 267), (271, 273), (278, 280),
     (286, 288), (292, 294), (299, 301), (306, 309), (313, 316), (320, 322), (326, 328), (327, 330), (334, 336),
     (341, 343), (348, 350), (355, 358), (362, 365), (369, 371), (376, 379), (383, 385), (389, 391), (390, 393)],
    [(130, 237)], [(103, 147)], [(18, 44), (97, 301), (284, 293)], [(53, 67), (57, 122), (206, 251)],
    [(2, 5), (9, 12), (17, 19), (23, 26), (30, 32), (31, 33), (44, 47), (51, 54), (58, 60), (59, 61), (65, 68),
     (72, 75), (79, 82), (83, 85), (86, 88), (87, 89), (93, 95), (94, 96), (100, 102), (101, 103), (107, 109),
     (108, 110), (109, 111), (114, 117), (119, 121), (121, 123), (122, 124), (128, 131), (135, 138), (143, 145),
     (150, 152), (205, 207), (220, 222), (226, 228), (233, 235), (234, 236), (240, 242), (241, 243), (254, 257),
     (261, 263), (262, 264), (268, 270), (269, 271), (275, 277), (276, 278), (289, 291), (290, 292), (296, 298),
     (297, 299), (303, 305), (304, 306), (310, 313), (318, 320), (324, 327), (331, 333), (332, 334), (338, 341),
     (345, 348), (352, 354), (358, 360), (360, 362), (367, 369), (373, 375), (374, 376), (381, 383), (387, 390)],
    [(118, 148)], [(218, 379)],
    [(0, 1), (2, 5), (9, 12), (16, 19), (23, 26), (30, 33), (37, 40), (43, 45), (44, 46), (45, 47), (47, 49), (48, 50),
     (50, 52), (51, 53), (52, 54), (58, 61), (65, 68), (72, 75), (79, 82), (86, 89), (93, 96), (100, 102), (101, 103),
     (107, 109), (108, 110), (109, 111), (114, 117), (119, 121), (121, 123), (122, 124), (128, 130), (129, 131),
     (135, 138), (142, 144), (143, 145), (144, 146), (149, 152), (171, 173), (234, 236), (262, 264), (268, 271),
     (276, 278), (282, 285), (289, 292), (296, 299), (303, 305), (304, 306), (310, 313), (317, 319), (318, 320),
     (324, 327), (331, 334), (338, 341), (345, 348), (352, 355), (356, 358), (358, 360), (360, 362), (364, 366),
     (366, 369), (372, 374), (373, 376), (380, 382), (381, 383), (387, 390), (394, 395)], [(41, 125), (268, 361)],
    [(96, 286)], [(61, 79), (65, 105), (241, 243)],
    [(2, 5), (9, 12), (16, 19), (23, 26), (30, 33), (37, 40), (44, 47), (51, 54), (58, 61), (66, 68), (73, 75),
     (79, 82), (81, 83), (86, 88), (88, 90), (93, 96), (95, 97), (101, 103), (107, 109), (108, 111), (114, 117),
     (121, 124), (129, 131), (135, 138), (141, 144), (143, 145), (149, 152), (156, 159), (163, 166), (171, 173),
     (177, 180), (184, 187), (191, 194), (198, 201), (205, 208), (212, 215), (220, 223), (226, 229), (233, 236),
     (235, 237), (240, 243), (247, 250), (254, 256), (256, 258), (261, 264), (268, 271), (275, 278), (277, 279),
     (282, 285), (289, 292), (297, 299), (304, 306), (311, 313), (317, 320), (319, 321), (324, 327), (331, 334),
     (338, 341), (345, 348), (353, 355), (367, 369), (373, 376), (380, 383), (386, 389), (388, 391), (394, 395)],
    [(11, 79), (100, 156), (274, 297)], [(115, 150)], [(127, 297)], [(116, 145)], [(225, 350)],
    [(114, 126), (113, 163)], [(112, 147)], [(11, 71), (132, 148), (274, 290)], [(288, 395)], [(9, 11), (194, 224)],
    [(213, 332)], [(61, 227)], [(68, 145), (251, 275), (372, 389)], [(159, 384)], [(130, 150), (360, 362)],
    [(27, 90), (16, 263)]
]

ORIGINAL_EVENT_KEYWORDS = [
    ['Soča', 'Soči'],
    ['Citroen', 'jednotka'],
    ['sériový', 'motor'],
    ['čtyřdveřový', 'objem'],
    ['naftový', 'charakterizovat'],
    ['vůz', 'vývoj'],
    ['přeplňovaný', 'hospodárný'],
    ['Charlie', 'Hebdo'],
    ['tvarování', 'laminární'],
    ['Civic', 'proudění'],
    ['model', 'start'],
    ['šarm', 'citroen'],
    ['Tourer', 'záď'],
    ['kupé', 'výstižně'],
    ['dCe', 'maxima'],
    ['downsizing', 'turbodmychadlo'],
    ['turba', 'dvoulitrový', 'nanometr'],
    ['odvodit', 'prostorný'],
    ['Twin', 'čtyřválec'],
    ['Turba', 'oceňovat'],
    ['Renault', 'norma'],
    ['výfuk', 'Energy'],
    ['Mans', 'brzdový', 'pedál'],
    ['vyšlápnutí', 'Pionýr'],
    ['dvojice', 'předpokládat'],
    ['hnací', 'prodlužovat'],
    ['vodíkový', 'zredukovat'],
    ['Championship', 'otálet', 'dotyk'],
    ['luxus', 'zachování'],
    ['stuttgartský', 'spojka'],
    ['vypínat', 'zdokonalovat'],
    ['točivý', 'World'],
    ['ETH', 'Climawork'],
    ['komercializovat', 'zpozornět'],
    ['vytrvalostní', 'Daimler'],
    ['lyžování', 'lyžař'],
    ['curyšský', 'pospíšit'],
    ['zastavení', 'fáze'],
    ['doplnění', 'Porsche'],
    ['nazutý', 'Ostřihom'],
    ['ližina', 'dveřní'],
    ['zavazadelník', 'Peugeot'],
    ['světlost', 'crossover'],
    ['podběh', 'trojúhelníkový'],
    ['Transporter', 'stěhovat'],
    ['navýšený', 'zorný', 'decentně'],
    ['Octavio', 'akumulátor'],
    ['celohliníkový', 'tříválec'],
    ['sloupec', 'převis'],
    ['punc', 'říše'],
    ['kompakt', 'elektromotor'],
    ['připojovat', 'terénní'],
    ['Cross', 'repertoár'],
    ['osudový', 'velkolepý'],
    ['výbava', 'stopa'],
    ['vstřik', 'PureTec'],
    ['splňovat', 'vylepšit'],
    ['THP', 'DOHC'],
    ['prototyp', 'zadní'],
    ['sací', 'zvětšený'],
    ['Ukrajina', 'ukrajinský'],
    ['Rusko', 'ruský', 'Rus'],
    ['bod', 'domácí'],
    ['série', 'základní'],
    ['volební', 'volič'],
    ['hlas', 'strana'],
    ['třída', 'technický'],
    ['povinný', 'velikost'],
    ['Praha', 'pražský'],
    ['událost', 'oslava'],
    ['host', 'Plzeň'],
    ['svět', 'světový'],
    ['Čína', 'Peking'],
    ['výsledek', 'vést'],
    ['útok', 'teroristický'],
    ['plyn', 'výroba'],
    ['střední', 'strategie'],
    ['Sparta', 'autor'],
    ['typický', 'standardní'],
    ['provoz', 'nízký'],
    ['spolupráce', 'rámec', 'akce'],
    ['slavit', 'získat'],
    ['redukce', 'poněkud'],
    ['zúčastnit', 'maďarský'],
    ['zavést', 'program'],
    ['Evropa', 'jednička'],
    ['drát', 'rozšířený'],
    ['modernizace', 'myšlenka'],
    ['čerpací', 'revoluční'],
    ['inovativní', 'vybudování'],
    ['dvouciferný', 'snový'],
    ['Izrael', 'izraelský'],
    ['úplný', 'drahý'],
    ['dítě', 'škola'],
    ['ručně', 'plast'],
    ['reprezentovat', 'obdobný'],
    ['posunout', 'předpis'],
    ['vývojový', 'infrastruktura']
]

ORIGINAL_EVENT_BURSTS = [
    [(21, 53)], [(113, 159)], [(120, 160)], [(118, 160)], [(119, 148)], [(115, 166)], [(115, 160)], [(372, 387)],
    [(105, 149)], [(106, 149)], [(106, 166)], [(116, 146)], [(106, 149)], [(116, 146)], [(115, 141)], [(115, 141)],
    [(114, 142)], [(116, 146)], [(115, 142)], [(115, 137)], [(115, 138)], [(116, 143)], [(116, 134)], [(116, 134)],
    [(119, 148)], [(116, 134)], [(116, 134)], [(116, 134)], [(116, 143)], [(116, 134)], [(116, 134)], [(116, 134)],
    [(104, 132)], [(104, 132)], [(89, 177)], [(0, 331)], [(105, 132)], [(115, 133)], [(116, 134)], [(131, 150)],
    [(131, 151)], [(101, 125)], [(131, 151)], [(131, 151)], [(131, 150)], [(131, 151)], [(143, 165)], [(103, 127)],
    [(131, 150)], [(131, 150)], [(143, 165)], [(131, 150)], [(131, 150)], [(131, 150)], [(114, 137)], [(102, 123)],
    [(116, 134)], [(102, 123)], [(110, 185)], [(102, 124)],
    [(53, 67), (57, 122), (206, 251)], [(52, 101), (207, 249), (328, 372)], [(31, 119), (274, 358)],
    [(78, 144), (360, 362)], [(11, 79), (101, 156), (274, 298)], [(40, 148), (269, 332)], [(118, 160), (319, 321)],
    [(113, 135), (144, 167)], [(40, 124), (257, 330)], [(91, 171), (304, 363)], [(31, 122), (251, 352)],
    [(75, 196), (359, 392)], [(108, 160), (286, 331)], [(65, 145), (266, 365)], [(109, 392), (372, 387)],
    [(108, 158), (265, 323)], [(114, 128), (129, 152)], [(30, 128), (235, 336)], [(114, 128), (131, 152)],
    [(119, 158), (341, 388)], [(102, 151), (268, 354)], [(66, 167), (268, 334)], [(115, 132), (138, 153)],
    [(130, 151), (298, 322)], [(105, 155), (302, 395)], [(128, 161), (226, 244), (319, 395)], [(116, 133), (136, 156)],
    [(104, 130), (120, 149)], [(115, 133), (294, 296)], [(114, 123), (128, 136)], [(128, 135), (140, 152)],
    [(120, 269), (325, 375)], [(114, 117), (116, 141)], [(87, 166), (264, 387)], [(128, 135), (140, 152)],
    [(114, 117), (117, 137)], [(114, 124), (128, 136)], [(115, 123), (128, 136)]
]

CLUSTER_LABELS = [
    'Ukrajina', 'Rusko', 'policie', 'soud', 'Zeman', 'EU', 'Sparta', 'festival', 'Babiš', 'Putin', 'Google',
    'ekonomika', 'letadlo', 'východ', 'politika', 'zabít', 'poslanec', 'armáda', 'Kyjev', 'Škoda', 'hokejista',
    'fotbalista', 'doprava', 'vražda', 'Vánoce', 'Francie', 'sport', 'NATO', 'Moskva', 'ropa', 'turnaj', 'Obama',
    'referendum', 'ebola', 'parlament', 'koalice', 'Paříž', 'automobil', 'mistrovství', 'elektrárna', 'Sýrie',
    'islamista', 'Brusel', 'olympiáda', 'sníh', 'průmysl', 'revoluce', 'výbuch', 'finance', 'terorista'
]


def automatic_prf_matching(event_keywords, event_bursts):
    real_events_path = '../../real_events'
    real_events = []
    evaluated_events = []

    for file in os.listdir(real_events_path):
        with open(os.path.join(real_events_path, file), 'r', encoding='utf8') as f:
            real_event = []
            for line in f:
                split = line.split('\t')

                if len(split) != 3:
                    continue

                lemma = split[1]
                pos = split[2]

                if pos[0] in 'NVAD':
                    real_event.append(lemma)

        real_events.append((int(file.split('.')[0]), frozenset(real_event)))

    for i, (bursts, keywords) in enumerate(zip(event_bursts, event_keywords)):
        set_keywords = frozenset(keywords)
        real_event_id = -1

        for j, (real_day, real_keywords) in enumerate(real_events):
            if len(set_keywords & real_keywords) > 0:
                if any(burst_start <= real_day <= burst_end for burst_start, burst_end in bursts):
                    real_event_id = j
                    break

        evaluated_events.append((i, real_event_id))

    return evaluated_events, [1 for _ in set(event_pair[1] for event_pair in evaluated_events)]


def events_prf(events, events_recall):
    good_events = sum(1 for _ in filter(lambda event: event[1] > -1, events))

    precision = good_events / len(events)
    recall = sum(events_recall) / len(REAL_EVENTS)
    f_measure = (2 * precision * recall) / (precision + recall)

    return precision, recall, f_measure


def events_redundancy(events):
    unique_events = len(events)
    total_events = sum(map(len, events))
    return 1 - unique_events / total_events


def events_noisiness(events):
    noisy_events = len(events[0])
    good_events = len(events[1])
    total_events = noisy_events + good_events
    return noisy_events / total_events


def cluster_purity(documents_path):
    with open(documents_path, 'rb') as f:
        event_documents = pickle.load(f)

    total_tagged = 0
    numerator = 0

    for i, event in enumerate(event_documents):
        keyword_counter = {keyword: 0 for keyword in CLUSTER_LABELS}
        seen = set()

        # if i < 60:
        #     print('\nEvent {:d}'.format(i))

        for burst_start, burst_end, burst_docs in event:
            for j, document in enumerate(burst_docs):
                # if i < 60 and j < 10:
                #     print(document.document.name_lemma)
                #     print(document.document.sentences_lemma)
                #     print()
                doc_name = document.document.name_lemma

                if tuple(doc_name) in seen:
                    continue

                seen.add(tuple(doc_name))

                # print(document.document.name_forms)

                for token in doc_name:
                    if token in keyword_counter:
                        keyword_counter[token] += 1

        keyword_counter = collections.Counter(keyword_counter)
        most_common, frequency = keyword_counter.most_common(1)[0]
        event_tagged = sum(keyword_counter.values())
        total_tagged += event_tagged

        event_purity = frequency / event_tagged if event_tagged > 0 else 0
        numerator += event_purity * event_tagged

    return numerator / total_tagged


def format_prf(method, dps, num_events, *prf):
    print('Detection method: {:s}'.format(method))
    print('DPS boundary: {:.02f}'.format(dps))
    print('Events found: {:d}'.format(num_events))
    print('Precision: {:.02f}%, Recall: {:.02f}%, F-measure: {:.02f}'.format(prf[0] * 100, prf[1] * 100, prf[2]))
    print()


def format_redundancy(method, dps, redundancy):
    print('Detection method: {:s}'.format(method))
    print('DPS boundary: {:.02f}'.format(dps))
    print('Detection redundancy: {:.02f}%'.format(redundancy * 100))
    print()


def format_noisiness(method, dps, noisiness):
    print('Detection method: {:s}'.format(method))
    print('DPS boundary: {:.02f}'.format(dps))
    print('Detection noisiness: {:.02f}%'.format(noisiness * 100))
    print()


def format_purity(method, purity):
    print('Detection method: {:s}'.format(method))
    print('Document cluster purity: {:.02f}%'.format(purity * 100))
    print()


def main():
    PICKLE_PATH = '../event_detection/pickle'
    EVENT_SUMM_DOCS_CLUSTERS_PATH = os.path.join(PICKLE_PATH, 'event_summ_docs_clusters.pickle')
    EVENT_SUMM_DOCS_GREEDY_PATH = os.path.join(PICKLE_PATH, 'event_summ_docs_greedy.pickle')
    EVENT_SUMM_DOCS_ORIGINAL_PATH = os.path.join(PICKLE_PATH, 'event_summ_docs_original.pickle')

    # print('CLUSTERS')
    # format_purity('Clusters', cluster_purity(EVENT_SUMM_DOCS_CLUSTERS_PATH))
    # print()
    # print('ORIGINAL')
    # format_purity('Original', cluster_purity(EVENT_SUMM_DOCS_ORIGINAL_PATH))
    # exit()

    print('Precision, Recall, F-measure AUTOMATIC')
    print('-' * 20)
    automatic_precision, automatic_recall = automatic_prf_matching(CLUSTERS_EVENT_KEYWORDS, CLUSTERS_EVENT_BURSTS)
    format_prf('Clusters', 0.01, len(CLUSTERS_EVENT_KEYWORDS), *events_prf(automatic_precision, automatic_recall))
    automatic_precision, automatic_recall = automatic_prf_matching(GREEDY_EVENT_KEYWORDS, GREEDY_EVENT_BURSTS)
    format_prf('Greedy', 0.04, len(GREEDY_EVENT_KEYWORDS), *events_prf(automatic_precision, automatic_recall))
    automatic_precision, automatic_recall = automatic_prf_matching(ORIGINAL_EVENT_KEYWORDS, ORIGINAL_EVENT_BURSTS)
    format_prf('Original', 0.05, len(ORIGINAL_EVENT_KEYWORDS), *events_prf(automatic_precision, automatic_recall))

    print('Precision, Recall, F-measure')
    print('-' * 20)

    format_prf('Clusters', 0.01, len(CLUSTERS_EVENT_KEYWORDS), *events_prf(CLUSTERS_PRECISION, CLUSTERS_RECALL))
    format_prf('Greedy', 0.04, len(GREEDY_EVENT_KEYWORDS), *events_prf(GREEDY_PRECISION, GREEDY_RECALL))
    format_prf('Original', 0.05, len(ORIGINAL_EVENT_KEYWORDS), *events_prf(ORIGINAL_PRECISION, ORIGINAL_RECALL))

    print('Redundancy')
    print('-' * 20)
    for method, dps, events in EVENTS_REDUNDANCY:
        format_redundancy(method, dps, events_redundancy(events))

    print('Noisiness')
    print('-' * 20)
    for method, dps, events in EVENTS_NOISINESS:
        format_noisiness(method, dps, events_noisiness(events))


if __name__ == '__main__':
    main()
