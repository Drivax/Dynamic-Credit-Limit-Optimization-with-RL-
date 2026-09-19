# Rapport des expériences PD longitudinales

## État initial et modifications

L'audit complet est dans [pd_audit.md](pd_audit.md). Le simulateur disposait déjà de trajectoires de 24 mois, d'un défaut absorbant et d'une séparation entre observations et traits cachés. Le modèle de risque était entraîné sur des snapshots synthétiques indépendants à un mois ; il ne validait pas une PD longitudinale. Aucun coefficient du DGP n'a été modifié.

**CREATED** : `configs/pd_model.yaml` ; modules risk `features`, `dataset`, `generation`, `settings`, `longitudinal`, `calibration`, `evaluation`, `reporting` ; `experiments/train_pd.py`, `experiments/pd_env_smoke.py` ; `tests/test_longitudinal_pd.py` ; `docs/pd_audit.md`, `docs/pd_model.md`, ce rapport ; artefacts PD, CSV/JSON et six figures.

**MODIFIED** : environnement et interface PD ; `experiments/train_ppo.py` ; README principal et README des expériences ; documentation environnement/DGP ; `.gitignore` pour conserver les petites tables et figures utilisées dans le README.

**MOVED** : aucun. **DELETED** : aucun. Les composants de simulation et diagnostics indépendants sont conservés.

## Cible, features et frontières

Pour un client actif à t : **Y(i,t,H)=1{t < T_i ≤ t+H}**, H=12 mois configurable. L'observation est l'état clôturé à t, avant la décision et les événements du mois suivant. Le défaut est le premier événement Bernoulli du DGP, pas un seuil juridique de retard.

Les candidats doivent satisfaire t+H ≤ fin administrative prévue, indépendamment de leur futur label. Un défaut observé dans cette fenêtre résout Y=1 ; un zéro exige un suivi effectif jusqu'à t+H. Les fenêtres inconnues ne deviennent jamais des zéros. Les clients déjà en défaut sont exclus. Sur ce run, aucune censure imprévue n'est présente ; les fins administratives sont comptabilisées explicitement.

Les snapshots n'ont pas le même dénominateur que les événements de défaut : un seul défaut peut rendre positifs plusieurs snapshots antérieurs.

**Features observables courantes** : `credit_limit`, `balance`, `utilization`, `income`, `payment_ratio`, `months_delinquent`, `behavioral_score`, `tenure_months`, `monthly_spend`, `late_payments_6m`, `income_log_change`, `macro_income_growth`, `macro_spending_growth`, `macro_credit_stress`.

**Features dérivées/historiques** : `history_months`, `debt_to_income`, `utilization_lag1`, `utilization_mean3`, `utilization_max6`, `payment_mean3`, `payment_min6`, `income_volatility6`, `balance_change1`, `income_change3`, `limit_change6`.

**Variables interdites** : `adverse_income_event`, `creditworthiness`, `income_stability`, `initial_income`, `initial_score`, `p_default_true`, `payment_propensity`, `spending_elasticity`, `spending_propensity`, `true_default_probability`, `true_pd`, ainsi que `latent_*`, `shock_*`, `future_*`, `next_*`, labels, actions/rewards et toute colonne non explicitement autorisée.

Les moyennes/min/max/volatilités portent sur t−k,...,t−1 ; les valeurs courantes sont séparées. Les lags absents restent manquants, puis sont imputés uniquement à partir du train avec indicateurs de manque. Aucune observation n'est supprimée pour manque d'historique. Les features en ligne et hors ligne utilisent le même constructeur.

## Population, split et calibration

Le train, la validation, la calibration, le test et l'OOT comportent 6 100 clients distincts. Leurs cohortes commencent respectivement aux mois 0,25,50,75,100. Les observations admissibles s'étendent jusqu'à +12 ; les labels finissent au plus tard à +24. Les identités sont disjointes et toutes les fenêtres de labels sont arrivées à maturité avant la cohorte suivante.

Le calendrier macro commun est réellement utilisé dans les transitions. L'initialisation des cohortes reste stationnaire ; il ne s'agit pas d'une preuve de robustesse à toute dérive démographique. Aucun mois de stress n'est présent dans le train de ce run. La validation contient uniquement l'expansion ; la calibration, le test et l'OOT voient d'autres segments du calendrier, dont du stress.

La validation sert de diagnostic sans recherche d'hyperparamètres. Seuls les modèles bruts y sont évalués : appliquer le calibrateur appris plus tard aurait donné une évaluation rétrospective. La calibration est exclusivement entraînée sur sa cohorte dédiée ; le test et l'OOT ne sont jamais utilisés pour le fit ni pour sélectionner un gagnant.

Les six expériences de scénario/policy rejouent les mêmes 1 000 clients OOT avec les mêmes états initiaux, traits cachés et chocs. Ces lignes ne sont pas six populations indépendantes. La behavior policy tire les cinq actions avec probabilités 0,1/0,2/0,4/0,2/0,1 sous les bornes existantes.

| Échantillon | Clients | Snapshots | Snapshots positifs | Prévalence | Événements défaut sur épisode | Période observée |
|---|---:|---:|---:|---:|---:|---|
| train | 2500 | 24941 | 10824 | 43.40% | 1597 | 0–12 |
| validation | 800 | 8046 | 3371 | 41.90% | 497 | 25–37 |
| calibration | 800 | 7560 | 3696 | 48.89% | 548 | 50–62 |
| test | 1000 | 10050 | 4603 | 45.80% | 702 | 75–87 |
| oot | 1000 | 7359 | 4170 | 56.67% | 776 | 100–112 |
| baseline_behavior | 1000 | 9891 | 4306 | 43.53% | 643 | 100–112 |
| mild_stress_behavior | 1000 | 9439 | 5010 | 53.08% | 705 | 100–112 |
| severe_stress_behavior | 1000 | 7896 | 6062 | 76.77% | 894 | 100–112 |
| baseline_static | 1000 | 9922 | 4377 | 44.11% | 639 | 100–112 |
| baseline_always_increase | 1000 | 10429 | 3811 | 36.54% | 563 | 100–112 |
| baseline_always_decrease | 1000 | 8790 | 4719 | 53.69% | 710 | 100–112 |

## Modèles réellement entraînés

Baseline constante = prévalence des snapshots train, **43,3984%**. Régression logistique L2, C=1, imputation/missing indicators et standardisation. HistGradientBoosting sklearn : 150 itérations, 15 feuilles, L2=1, sans split interne d'early stopping. Pour chacun, variante brute et calibration sigmoid(a+b·logit(PD)) sur la cohorte de calibration. Pas de pondération des classes, SMOTE, feature selection ou tuning sur le test.

Le modèle exposé par défaut est la logistique calibrée, choix de référence pré-déclaré et configurable. Les résultats ne justifient pas un gagnant universel : le boosting calibré est meilleur en OOT mais nettement moins bien calibré sous stress sévère. Les coefficients logistiques standardisés sont exportés ; leur signe conditionnel en présence de variables corrélées n'est pas un effet causal.

## Table complète des performances

Valeur centrale **[IC 95%]**. Bootstrap percentile de **200 tirages de clients entiers**, avec multiplicité et conservation des dépendances intra-client. Les IC conditionnent sur le modèle entraîné et le calendrier macro réalisé ; ils ne couvrent pas l'incertitude de réentraînement ou des scénarios macro. PR-AUC désigne l'average precision. Les prévalences et tailles sont dans la table précédente.

| Échantillon | Modèle | ROC-AUC [IC] | PR-AUC [IC] | Brier [IC] | Log loss [IC] |
|---|---|---|---|---|---|
| validation | constant | 0.500 [0.500, 0.500] | 0.419 [0.385, 0.449] | 0.244 [0.239, 0.248] | 0.680 [0.671, 0.688] |
| validation | logistic | 0.843 [0.825, 0.862] | 0.807 [0.780, 0.831] | 0.158 [0.149, 0.167] | 0.476 [0.451, 0.500] |
| validation | boosting | 0.840 [0.818, 0.860] | 0.809 [0.782, 0.834] | 0.159 [0.148, 0.169] | 0.480 [0.451, 0.508] |
| test | constant | 0.500 [0.500, 0.500] | 0.458 [0.424, 0.488] | 0.249 [0.244, 0.253] | 0.691 [0.682, 0.699] |
| test | logistic | 0.842 [0.825, 0.859] | 0.823 [0.797, 0.846] | 0.162 [0.153, 0.171] | 0.487 [0.463, 0.510] |
| test | logistic_calibrated | 0.842 [0.825, 0.859] | 0.823 [0.797, 0.846] | 0.162 [0.153, 0.171] | 0.486 [0.463, 0.510] |
| test | boosting | 0.846 [0.829, 0.866] | 0.831 [0.806, 0.855] | 0.161 [0.150, 0.170] | 0.485 [0.457, 0.509] |
| test | boosting_calibrated | 0.846 [0.829, 0.866] | 0.831 [0.806, 0.855] | 0.160 [0.149, 0.170] | 0.483 [0.454, 0.507] |
| oot | constant | 0.500 [0.500, 0.500] | 0.567 [0.535, 0.599] | 0.263 [0.259, 0.267] | 0.720 [0.711, 0.728] |
| oot | logistic | 0.845 [0.822, 0.860] | 0.875 [0.850, 0.892] | 0.163 [0.153, 0.175] | 0.487 [0.464, 0.518] |
| oot | logistic_calibrated | 0.845 [0.822, 0.860] | 0.875 [0.850, 0.892] | 0.172 [0.161, 0.186] | 0.511 [0.483, 0.548] |
| oot | boosting | 0.845 [0.822, 0.864] | 0.875 [0.852, 0.896] | 0.161 [0.151, 0.172] | 0.484 [0.460, 0.514] |
| oot | boosting_calibrated | 0.845 [0.822, 0.864] | 0.875 [0.852, 0.896] | 0.158 [0.148, 0.170] | 0.478 [0.453, 0.510] |
| baseline_behavior | constant | 0.500 [0.500, 0.500] | 0.435 [0.408, 0.465] | 0.246 [0.242, 0.250] | 0.685 [0.677, 0.693] |
| baseline_behavior | logistic | 0.827 [0.806, 0.842] | 0.801 [0.769, 0.822] | 0.167 [0.160, 0.178] | 0.500 [0.482, 0.526] |
| baseline_behavior | logistic_calibrated | 0.827 [0.806, 0.842] | 0.801 [0.769, 0.822] | 0.171 [0.163, 0.182] | 0.507 [0.487, 0.533] |
| baseline_behavior | boosting | 0.835 [0.817, 0.853] | 0.814 [0.787, 0.836] | 0.163 [0.154, 0.174] | 0.492 [0.469, 0.519] |
| baseline_behavior | boosting_calibrated | 0.835 [0.817, 0.853] | 0.814 [0.787, 0.836] | 0.168 [0.158, 0.179] | 0.503 [0.478, 0.528] |
| mild_stress_behavior | constant | 0.500 [0.500, 0.500] | 0.531 [0.502, 0.561] | 0.258 [0.255, 0.262] | 0.710 [0.703, 0.718] |
| mild_stress_behavior | logistic | 0.828 [0.811, 0.845] | 0.843 [0.818, 0.861] | 0.171 [0.163, 0.179] | 0.509 [0.489, 0.529] |
| mild_stress_behavior | logistic_calibrated | 0.828 [0.811, 0.845] | 0.843 [0.818, 0.861] | 0.169 [0.160, 0.177] | 0.503 [0.482, 0.525] |
| mild_stress_behavior | boosting | 0.843 [0.827, 0.861] | 0.859 [0.836, 0.878] | 0.168 [0.160, 0.179] | 0.504 [0.480, 0.531] |
| mild_stress_behavior | boosting_calibrated | 0.843 [0.827, 0.861] | 0.859 [0.836, 0.878] | 0.162 [0.153, 0.170] | 0.488 [0.463, 0.511] |
| severe_stress_behavior | constant | 0.500 [0.500, 0.500] | 0.768 [0.735, 0.797] | 0.290 [0.285, 0.294] | 0.773 [0.764, 0.781] |
| severe_stress_behavior | logistic | 0.807 [0.775, 0.835] | 0.928 [0.911, 0.944] | 0.181 [0.172, 0.191] | 0.531 [0.509, 0.557] |
| severe_stress_behavior | logistic_calibrated | 0.807 [0.775, 0.835] | 0.928 [0.911, 0.944] | 0.164 [0.155, 0.174] | 0.489 [0.465, 0.518] |
| severe_stress_behavior | boosting | 0.848 [0.821, 0.872] | 0.945 [0.931, 0.957] | 0.218 [0.205, 0.230] | 0.640 [0.602, 0.671] |
| severe_stress_behavior | boosting_calibrated | 0.848 [0.821, 0.872] | 0.945 [0.931, 0.957] | 0.186 [0.175, 0.197] | 0.552 [0.522, 0.580] |
| baseline_static | constant | 0.500 [0.500, 0.500] | 0.441 [0.415, 0.473] | 0.247 [0.243, 0.251] | 0.686 [0.679, 0.695] |
| baseline_static | logistic | 0.827 [0.806, 0.845] | 0.801 [0.772, 0.827] | 0.168 [0.160, 0.178] | 0.501 [0.480, 0.527] |
| baseline_static | logistic_calibrated | 0.827 [0.806, 0.845] | 0.801 [0.772, 0.827] | 0.172 [0.162, 0.182] | 0.509 [0.484, 0.537] |
| baseline_static | boosting | 0.839 [0.819, 0.856] | 0.819 [0.789, 0.843] | 0.162 [0.153, 0.173] | 0.488 [0.465, 0.517] |
| baseline_static | boosting_calibrated | 0.839 [0.819, 0.856] | 0.819 [0.789, 0.843] | 0.167 [0.157, 0.178] | 0.499 [0.474, 0.527] |
| baseline_always_increase | constant | 0.500 [0.500, 0.500] | 0.365 [0.338, 0.396] | 0.237 [0.233, 0.241] | 0.666 [0.659, 0.674] |
| baseline_always_increase | logistic | 0.834 [0.814, 0.853] | 0.765 [0.732, 0.795] | 0.162 [0.153, 0.171] | 0.487 [0.464, 0.512] |
| baseline_always_increase | logistic_calibrated | 0.834 [0.814, 0.853] | 0.765 [0.732, 0.795] | 0.172 [0.163, 0.182] | 0.511 [0.487, 0.538] |
| baseline_always_increase | boosting | 0.837 [0.819, 0.859] | 0.780 [0.746, 0.807] | 0.160 [0.149, 0.171] | 0.484 [0.455, 0.510] |
| baseline_always_increase | boosting_calibrated | 0.837 [0.819, 0.859] | 0.780 [0.746, 0.807] | 0.173 [0.162, 0.184] | 0.515 [0.486, 0.542] |
| baseline_always_decrease | constant | 0.500 [0.500, 0.500] | 0.537 [0.510, 0.572] | 0.259 [0.256, 0.264] | 0.712 [0.705, 0.721] |
| baseline_always_decrease | logistic | 0.859 [0.842, 0.874] | 0.878 [0.858, 0.896] | 0.163 [0.154, 0.172] | 0.491 [0.470, 0.518] |
| baseline_always_decrease | logistic_calibrated | 0.859 [0.842, 0.874] | 0.878 [0.858, 0.896] | 0.155 [0.147, 0.164] | 0.473 [0.450, 0.496] |
| baseline_always_decrease | boosting | 0.840 [0.820, 0.857] | 0.864 [0.840, 0.883] | 0.173 [0.164, 0.183] | 0.516 [0.494, 0.541] |
| baseline_always_decrease | boosting_calibrated | 0.840 [0.820, 0.857] | 0.864 [0.840, 0.883] | 0.164 [0.156, 0.174] | 0.495 [0.475, 0.517] |

## Calibration, stress et décalage induit par les policies

Le test affiche une AUC autour de 0,84 et un Brier autour de 0,16 contre 0,249 pour la constante. Il n'y a pas de performance quasi parfaite à expliquer par une variable cachée injectée.

La calibration ne résout pas tous les changements de distribution. En OOT, la logistique calibrée prédit environ 67,3% contre 56,7% de positifs, et son Brier se dégrade de 0,163 brut à 0,172 calibré. Le boosting calibré y prédit environ 57,7% et obtient un Brier de 0,158.

Sous stress sévère, le boosting calibré a une AUC de 0,848 mais une PD moyenne de 54,7% contre 76,8% observé : un bon classement n'empêche pas une forte sous-estimation du risque. La logistique calibrée sous-estime également le risque (62,8%), mais moins fortement. Le train sans stress ne permet pas d'apprendre directement le facteur credit_stress : son coefficient logistique est nul. Aucun changement du DGP ou des seeds n'a été effectué pour cacher cette limite.

La table suivante expose les conséquences des policies sur la logistique calibrée, sans réentraînement ni recalibration :

| Policy/scénario | Prévalence | PD moyenne | AUC | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| baseline_behavior | 43.53% | 48.40% | 0.827 | 0.171 | 0.053 |
| baseline_static | 44.11% | 49.08% | 0.827 | 0.172 | 0.052 |
| baseline_always_increase | 36.54% | 47.35% | 0.834 | 0.172 | 0.108 |
| baseline_always_decrease | 53.69% | 49.07% | 0.859 | 0.155 | 0.046 |

Une hausse systématique n'augmente pas nécessairement le défaut dans ce DGP : elle réduit aussi mécaniquement l'utilisation et évite les situations au-dessus de la limite. Ces comparaisons reflètent des changements de comportement et de survie. Elles ne disent pas quelle policy maximise le profit ni quelle décision appliquer dans une banque réelle. Les distributions de features sont exportées avec moyenne, q10, médiane, q90 et fréquence de manque ; les prédictions, déciles, sous-groupes et métriques par mois sont également exportés.


Distribution moyenne des états observés (snapshots admissibles) :

| Échantillon | Limite EUR | Balance EUR | Utilisation | Payment ratio |
|---|---:|---:|---:|---:|
| baseline_behavior | 6930 | 3908 | 0.593 | 0.273 |
| baseline_static | 6983 | 4034 | 0.584 | 0.273 |
| baseline_always_increase | 12198 | 4223 | 0.356 | 0.288 |
| baseline_always_decrease | 3115 | 2453 | 0.993 | 0.250 |
| severe_stress_behavior | 6974 | 3787 | 0.568 | 0.239 |

## Vérification de snapshots et trajectoires

Onze snapshots des trois premiers clients du test ont été inspectés et leurs labels/lag moyens recalculés à partir des lignes datées. Exemples :

| Client | t | Dernier mois historique | Défaut observé | Y à 12 mois | PD |
|---|---:|---:|---:|---:|---:|
| cohort3-0 | 0 | None | 18 | 0 | 0.656 |
| cohort3-0 | 3 | 2 | 18 | 0 | 0.826 |
| cohort3-0 | 6 | 5 | 18 | 1 | 0.657 |
| cohort3-0 | 12 | 11 | 18 | 1 | 0.719 |
| cohort3-1 | 0 | None | 7 | 1 | 0.884 |
| cohort3-1 | 3 | 2 | 7 | 1 | 0.986 |
| cohort3-1 | 6 | 5 | 7 | 1 | 0.997 |
| cohort3-2 | 0 | None | None | 0 | 0.267 |
| cohort3-2 | 3 | 2 | None | 0 | 0.780 |
| cohort3-2 | 6 | 5 | None | 0 | 0.128 |
| cohort3-2 | 12 | 11 | None | 0 | 0.054 |

Pour cohort3-0 à t=6, le défaut à 18 est exactement à la borne incluse t+12. La moyenne d'utilisation vaut 1,02963 et utilise uniquement les mois 3,4,5. Pour cohort3-2 à t=3, une mensualité manquée fait monter la PD à 0,780 alors qu'aucun défaut ne se réalise sur l'épisode : le modèle reste imparfait. Les courbes de trois trajectoires sont inspectées avec le hazard mensuel caché, explicitement marqué diagnostic uniquement et non comparable comme cible de calibration à la PD 12 mois.

## Environnement, tests et performances

L'environnement construit une observation explicite, conserve sept lignes au maximum et appelle `predict_history` avant chaque prochaine décision. Le score d'ouverture est celui du reward du mois suivant. Le DGP ne reçoit pas l'estimateur. L'historique est propre à chaque environnement et réinitialisé au reset. Le défaut terminal a un sentinel et n'est pas scoré comme un client actif.

**98 tests passent, 0 échoué.** Les nouveaux tests couvrent les bornes de fenêtre, censure, exclusion des fins administratives, clients disjoints et labels arrivés à maturité, refus des features interdites/nouvelles, mutation agressive du futur, historique par client, valeurs manquantes, imputer/scaler appris uniquement sur train, calibration interdite sur le test, probabilités valides, sauvegarde/rechargement, seeds, bootstrap par client, parité offline/online et isolation entre environnements partageant le modèle. Le test de graphique avec index non contigu protège contre une erreur d'alignement Pandas détectée à l'inspection visuelle.

Le smoke test de l'artefact dans l'environnement a exécuté **10 clients, 181 transitions**, avec parité offline/online à toutes les décisions actives. Le diagnostic DGP de 100 clients passe ses sept vérifications structurelles. Le smoke PPO à budget demandé de 256 transitions a été exécuté, sauvegardé et rechargé ; ce n'est pas une comparaison économique fiable.

Temps mesuré du run de référence : pipeline complet **184.9 s**, fit logistique **0.138 s**, boosting **0.911 s**. La génération et les bootstrap dominent le temps total. Mesures indicatives locales, CPU avec un thread natif pour les modèles.

| Modèle | Feature history + inférence (ms) | Débit batch (lignes/s) |
|---|---:|---:|
| logistic | 0.592 | 1,916,184 |
| logistic_calibrated | 0.740 | 1,741,647 |
| boosting | 1.610 | 203,739 |
| boosting_calibrated | 1.777 | 214,663 |

## Reproduction et documentation

Les commandes du README ont été exécutées avec `.venv/Scripts/python.exe` : installation editable avec les extras dev/experiments et `--no-build-isolation`, pytest, pipeline PD complet, évaluation seule, smoke environnement, diagnostic DGP et smoke PPO (extras RL déjà disponibles). Le README décrit le système actuel, ses équations, un diagramme Mermaid, les commandes, les résultats réellement mesurés, garanties et limites, sans récit historique. Les deux figures du README sont générées par le pipeline et conservées dans les exceptions ciblées du gitignore.

Deux exécutions complètes ont produit des données strictement identiques pour les onze datasets, les prédictions, les effectifs et les 43 lignes de métriques, intervalles inclus. La comparaison exclut seulement les temps d’exécution. Le contrôle est sauvegardé dans `outputs/results/pd/reproducibility_check.json`.

Les paramètres, seed streams, DGP version/config, populations, périodes, features et calibrateurs sont intégrés aux artefacts. Manifest, package versions et hash sources permettent de retrouver l'exécution. Les grands jeux de trajectoires et modèles restent des outputs locaux. Les tables de synthèse CSV/JSON sont conservées avec le dépôt.

## Limites restantes et priorités non implémentées

1. Évaluer plusieurs calendriers macro indépendants et un backtest glissant ; le bootstrap client ne mesure pas le risque macro commun.
2. Étendre les trajectoires afin de valider les prédictions à horizon complet pour les décisions au-delà du mois 12.
3. Tester sur des cohortes d'entraînement incluant plusieurs régimes, selon un protocole fixé avant évaluation et sans masquer le résultat présent.
4. Étudier des calibrateurs suivis dans le temps avec fenêtres d'apprentissage strictement antérieures et tests de stabilité par sous-groupe.
5. Formaliser la dépendance de PD à la policy future, puis évaluer des estimands conditionnés à l'action sans les confondre avec de la causalité identifiée.
6. Revoir les pénalités/capital proxies et seuils en fonction du véritable horizon PD, avant toute revendication économique RL.
7. Quantifier l'incertitude due aux traits cachés avec répétitions conditionnelles contrôlées ; ne pas utiliser le hazard réalisé comme pseudo-label.
8. Ajouter une approche adaptée à la censure informative si des données réelles ou du churn sont introduits.
9. Valider les hypothèses comportementales, coefficients macro, défaut et recouvrement sur des données empiriques gouvernées avant toute conclusion externe.

Le DGP, les données et les résultats restent synthétiques ; aucune preuve d'applicabilité bancaire ni de supériorité RL n'est revendiquée.
