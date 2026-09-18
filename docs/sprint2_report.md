# Sprint 2 — rapport technique

Exécution locale du 18 septembre 2026, DGP 2.0, seed 42, horizon 24 mois.
Les paramètres n'ont pas été ajustés pour favoriser une politique. Les sorties
complètes sont dans `outputs/results/dgp/` et les six figures dans `outputs/figures/dgp/`.

## A. Sprint 1 audit

Avant modification : commit `0b619da`, arbre propre, 42 tests passants. Le simulateur
suivait déjà un même client, avec trois traits cachés, remboursement/dépenses
stochastiques, défaut distinct de la PD, terminaison et comptabilité explicites.
Manquaient notamment revenu évolutif, corrélations latentes, troisième régime macro,
scénarios sérialisables, chocs indexés réutilisables et diagnostics de population.
L'adaptateur GB construisait encore un DataFrame par prédiction. Voir
[sprint2_audit.md](sprint2_audit.md). Générateur initial, entraînement GB, récompense,
baselines et archives ont été conservés.

## B. Files changed

**CREATED** : `simulation/{behavior,default,dgp,macro,shocks}.py`,
`envs/observation.py`, `evaluation/dgp_diagnostics.py` sous `src/credit_rl/` ;
`configs/macro_scenarios.yaml`, `experiments/dgp_sanity.py`,
`tests/test_dgp_v2.py`, `tests/test_macro_and_shocks.py`,
`docs/{dgp,sprint2_audit,sprint2_report,environment_v1}.md` et sorties DGP ignorées par Git.

**MODIFIED** : `config.py`, `simulation/customer.py`, `envs/credit_limit_env.py`,
`risk/pd_model.py`, `reward.py` (import uniquement), configuration simulation,
`experiments/{common,trajectory_sanity,train_ppo}.py`, deux anciens fichiers de tests,
`pyproject.toml`, README racine/experiments/data et `docs/environment.md`.

**MOVED** : aucun. L'ancienne spécification a été copiée en `environment_v1.md`.

**DELETED** : `simulation/dynamics.py`, remplacé par les modules DGP spécialisés.
Aucun artefact historique ni donnée utilisateur supprimé. Aucun commit/push effectué.

## C. DGP architecture

`customer` initialise état et traits persistants ; `macro` prépare les chemins
exogènes ; `shocks` fournit sept canaux indexés ; `behavior` réalise revenu,
paiement, dépenses et retards ; `default` calcule le hazard caché ; `dgp` orchestre
une transition pure sans RNG. `observation` impose la projection visible ; l'env
Gymnasium gère épisodes, récompense, PD et exports. Le DAG et les équations
exhaustives figurent dans [dgp.md](dgp.md).

## D. State definitions

**Observable** : mois, limite, principal, utilisation, revenu et variation logarithmique,
paiement/principal, dépenses récentes, score, retard courant, nombre de mois consécutifs
en retard, historique binaire de six mois, ancienneté, défaut réalisé, régime et trois
facteurs macro courants, PD imparfaite. Projection de 21 dimensions normalisées,
dont trois indicatrices de régime ; ID et ancres initiales ne sont pas des features.

**Latent** : creditworthiness z, spending propensity h, payment propensity p,
income stability k. Un facteur normal commun et quatre résidus induisent des
corrélations ; loadings (.70,-.25,.65,.65). Traits tirés une fois puis fixes.

**Exogenous** : chemin macro et chocs revenu normal/uniforme, dépenses normales,
paiement normal, incident uniforme, score normal, défaut uniforme. Les valeurs futures
sont cachées. Les ancres initiales de revenu/score sont du bookkeeping interne.

## E. Monthly event sequence

Observation et PD à t → décision → caps et limite effective → revenu → paiement sur
principal initial → retards → achats plafonnés au disponible → principal/score/historique
→ hazard sur état candidat avec macro de t → tirage du défaut → récompense → macro de
t+1 et observation suivante. Défaut absorbant prioritaire sur la troncature à H.

## F. Customer behavior equations

Notation : B principal, L limite, Y revenu, C dépenses, q ratio payé, d retards
consécutifs, z/h/p/k traits ; m/g/w = stress/croissance revenu/croissance dépenses.
Tous les chocs normaux sont bornés à ±6 ; σ est le lien logistique.

- `L'=clip(L*clip(A,.8,1.2),500,15000)` avec caps configurables.
- `v=1-k`, `sY=.025*(1+2v)*(1+.5m)`, `J=1[U<clip(.01+.06mv,0,1)]`.
  `δ=.08*log(Y0/Y)+g*(1+v)+sY*ε-sY²/2-.15J` ;
  `Y'=clip(Y*exp(clip(δ,-.35,.20)),300,30000)`.
- `q*=σ(logit(p)+.35z-.65B/L'-.30B/Y'-.65d-.60m+.40εP)`.
  `qbar=.5q+.5q*`. Incident si `Umiss<σ(-3+.8B/L'+.3B/Y'+.6d-.7z+.7m)` ;
  dans ce cas `qbar=min(qbar,.02)`. `P=min(B*qbar,.5Y')`, `q'=P/B` (1 si B=0).
- `d'=d+1` si B>0 et q'<.05, sinon 0 ; historique décalé et nouvel indicateur ajouté.
- `β=min(.15h,.60)` ; `demand=(.35C+.65*.35Y'h)*(1+w)*exp(βlog(L'/L)+.25εC-.25²/2)`.
  `C'=min(demand,max(0,L'-(B-P)))`.
- `B'=B-P+C'`, `u'=B'/L'`. Aucun intérêt capitalisé, aucune dette effacée par une baisse.

Les coefficients sont configurables ; calcul de demande/élasticité en log pour limiter
les débordements. Un retard désigne un paiement sous 5 % du principal, pas un DPD légal.

## G. Default DGP

Pour B'>0 :

`p_default_true = σ(-6 + 1.2B'/L' + .45B'/Y' + .65d' + .6(1-q') - .8z + .9m + 1.5max(0,-log(Y'/Y)))`.

Pour B'=0 : p=0. `default'=Udefault<p_default_true`. Le prochain état est absorbant
via la terminaison ; aucun tirage depuis le modèle de PD. Le hazard est conditionnel
aux comportements réalisés dans le mois, alors que la PD de décision précède ces
réalisations : leur nuage de points n'est pas une courbe de calibration.

## H. Macro process

Ordre expansion/normal/stress, matrice mensuelle :
`[[.88,.11,.01],[.035,.93,.035],[.01,.14,.85]]`.
Facteurs `(g,w,m)` respectifs : `(.003,.02,0)`, `(.001,0,0)`, `(-.01,-.12,1)`.

Baseline : normal constant. Mild : 1/4 normal, 1/4 stress ×.65, 1/2 normal.
Severe : 1/6 normal, 2/3 stress ×1.6, 1/6 normal.
Recovery : quatre quarts stress ×1.3, stress ×.6, normal, expansion.
Les fractions s'adaptent à H ; à H=24 le stress sévère couvre t=4…19.
Les facteurs affectent revenu, dépenses, paiement/retards et hazard. Les chemins
sauvegardés contiennent H+1 états, seul l'état courant étant visible.

## I. Action pathways

Limite → utilisation de début de mois → paiement/incident → retard et hazard ;
limite → élasticité et disponible → achats → principal/exposition → hazard et pertes.
Ces variables persistent au mois suivant. Les caps rendent la variation effective
parfois différente de la demande ; les deux sont exportées. Aucun coefficient direct
« grande limite donc défaut » ni avantage PPO ajouté.

## J. Randomness

SeedSequence combine seed, digest BLAKE2 stable de l'ID et digest du canal.
Le mois est l'index du tableau pré-généré. Allonger H conserve le préfixe ; réordonner
les clients ou terminer tôt une politique ne décale aucun autre tirage. Initialization
et macro ont leurs générateurs séparés. Les six comparaisons utilisent exactement
les mêmes états, traits et chocs individuels. Le diagnostic Markov utilise des chemins
indépendants par client ; les scénarios déterministes partagent un chemin de cohorte.

## K. Leakage controls

Traits, hazard, chocs et macro future absents des observations/info/historique public.
Le prédicteur ne reçoit que `ObservedRiskFeatures`. Export scientifique opt-in séparé
et table latente distincte. Données learnability collectées avant `step`, cible défaut
suivant, séparation par client sans jointure sur le diagnostic privilégié.
Tests : features autorisées, labels initiaux ignorés, indépendance du DGP vis-à-vis de
la PD, traits persistants, futurs chocs/macro sans fuite dans observation initiale,
CRN malgré défauts à des dates différentes. Python n'est pas une sandbox d'accès aux objets.

## L. Statistical diagnostics

Population Markov : **5 000 clients, 69 334 mois-clients à risque**. Hazard moyen
4,912 %, écart-type 11,282 %, médiane 1,187 %, P1 0,0703 %, P99 62,214 %.
Aucun hazard sous 10^-6 ou au-dessus de .999 : ni constant ni binaire.
Défauts mensuels observés par régime : expansion 4,378 %, normal 4,517 %, stress 9,561 %.
Les groupes de risque latent ordonnent le hazard moyen : 1,09 %, 2,90 %, 5,60 %, 9,09 %, 15,80 %.

Contrôles appariés à un mois (5 000 cas) : z diminué de 1 → +2,94 points de défaut ;
retard initial +2 → +4,50 points ; p remplacé par (p+1)/2 → paiement +214,04 EUR ;
h multiplié par 1,5 → achats +355,32 EUR. Les six contrôles directionnels passent.

Learnability : logistique simple, 3 500 clients train / 1 500 test, 48 646 / 20 688 lignes.
AUC **0,8751**, AP **0,4096**, Brier **0,03632** contre **0,04761** pour la constante ;
log-loss **0,13810** contre **0,19889**. Prévalence test 5,013 %. Il existe un signal
observable sans prédiction presque parfaite ; cela ne valide pas un modèle PD bancaire.

Six graphiques générés et inspectés : défaut par régime, utilisation temporelle,
retards temporels, dépenses/limite par propension, hazard/PD, dix trajectoires par
quintile latent. La baisse tardive des retards/utilisations reflète aussi la sélection
des survivants. L'association dépenses/limite ne mesure pas une élasticité causale.

## M. Macro stress results

Incidence sur 24 mois et moyennes par client (montants EUR ; rendement non actualisé) :

| Scénario statique | Défaut | Au moins un retard | Ratio payé moyen | Pertes | Rendement |
|---|---:|---:|---:|---:|---:|
| Baseline | 64,92 % | 78,10 % | 22,36 % | 2 126,45 | -1 579,03 |
| Mild | 70,82 % | 80,22 % | 21,18 % | 2 303,95 | -1 857,04 |
| Severe | 90,22 % | 85,26 % | 17,97 % | 2 773,68 | -2 599,56 |
| Recovery | 79,30 % | 81,04 % | 15,93 % | 2 327,56 | -2 104,02 |

Severe-baseline : +25,30 points de défaut, IC95 apparié [24,09 ; 26,51] ;
+647,23 EUR de pertes, IC95 [602,56 ; 691,90]. Le scénario recovery commence en stress :
il ne réanime pas les défauts passés. Ses moyennes sur durées actives ne constituent
pas une comparaison à exposition égale. Ces intervalles conditionnent sur une seule
configuration et les chemins imposés, pas sur l'incertitude du modèle macro.

## N. Policy-response sanity check

| Politique baseline | Défaut | Limite moyenne | Principal moyen | Achats mensuels moyens | Utilisation moyenne | Pertes | Rendement |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baisse répétée -20 % | 71,54 % | 2 664,35 | 2 569,93 | 455,12 | 111,85 % | 1 075,29 | -883,92 |
| Statique | 64,92 % | 6 930,74 | 4 430,11 | 1 052,20 | 65,43 % | 2 126,45 | -1 579,03 |
| Hausse répétée +20 % | 56,96 % | 12 725,24 | 4 824,14 | 1 229,18 | 39,47 % | 2 549,23 | -1 907,16 |

Les moyennes d'état sont calculées par client sur ses mois actifs puis moyennées.
Baisser la limite accroît la pression de remboursement/utilisation, mais comprime
achats et exposition ; cette réduction des pertes améliore ici le rendement.
Augmenter améliore utilisation/paiement et réduit l'incidence, mais accroît les pertes.
Hausse-statique : défaut -7,96 points, rendement -328,13 EUR (IC95 [-379,13 ; -277,13]).
Tous les rendements restent négatifs. Aucun entraînement ni classement de PPO dans ce sprint.

## O. Tests

**64 passés, 0 échoué** avec `.venv/Scripts/python.exe -m pytest -q`.
Comptabilité, contraintes/over-limit, défaut absorbant, seed/replay, absence de fuite,
PD indépendante, corrélations latentes, stabilité/revenu/stress, hazard directionnel,
élasticité hétérogène, sérialisation, caps effectifs, extrêmes numériques et absence
Pandas dans le hot path sont couverts. Config macro personnalisée respectée dès t=0.
Les checkers Gymnasium et SB3 passent ; les hashes du manifeste correspondent aux sources finales.

L'ancien `experiments.trajectory_sanity` adapté a aussi terminé : six clients,
quatre politiques, défaut 83,33 % dans chacun de ces petits groupes. Son client de
stress ciblé fait défaut dans 100 % des 200 essais, même en normal ; le rendement
normal est -5 302,65 contre -3 516,08 sous stress (durées/expositions différentes).
Ce diagnostic saturé n'est pas masqué et ne remplace pas la comparaison de population.

## P. Performance

Benchmark CPU mono-processus, 200 clients, 2 897 steps, PD logistique observable,
historique/diagnostic désactivés : **13 072 steps/s**, 0,222 s dans step et 0,268 s
reset inclus. Mesure finale dans `outputs/results/dgp/benchmark.json`.
Le débit dépend du matériel et exclut
reset, exports et entraînement. Les prédictions GB coûtent davantage. Le manifeste
fournit versions Python/paquets, paramètres résolus, seed et hashes des sources.

## Q. Limitations

Calibration absente et incidence cumulée très élevée ; init synthétique et historique
reconstruit ; retards simplifiés sans registre contractuel/collections ; LGD fixe ;
revenu ancré et borné ; réactions clients sans churn ni utilité ; macro observable
sans bruit ; proxy PD et GB historique mal adaptés ; biais de survivants dans les
moyennes actives ; une seed principale ; pas de valorisation terminale du portefeuille.
Les résultats établissent des mécanismes reproductibles, pas leur validité empirique,
ni une recommandation de crédit ou la supériorité d'une méthode d'optimisation.

## R. Next Sprint

1. Fixer des cibles empiriques et analyser la sensibilité des hazards, retards et durées.
2. Générer des panels PD avec observations à t, labels futurs et séparation clients/temps/scénarios.
3. Évaluer calibration et discrimination hors politique/scénario d'entraînement, avec censure explicite.
4. Répéter les comparaisons sur plusieurs seeds et chemins macro communs au portefeuille.
5. Clarifier valorisation terminale et arrears avant de comparer davantage de politiques.

Ces étapes ne sont pas implémentées. Aucun tuning PPO ou gros modèle PD ajouté.
