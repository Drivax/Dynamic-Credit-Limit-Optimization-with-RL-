# Audit réel du Sprint 1 avant Sprint 2

Point de départ : commit local `0b619da` (arbre propre). Lecture récursive des sources,
configurations, expériences, tests, README et spécification ; archives et sorties
historiques inventoriées. Aucun AGENTS.md ou notebook présent. Les **42 tests existants
passaient** avant modification (Python 3.12.14). Les données et résultats historiques
ne nécessitaient pas une nouvelle migration de répertoire.

| Dimension | Constat vérifié dans le code | Décision Sprint 2 |
|---|---|---|
| Longitudinalité | Snapshot CustomerState conservé puis remplacé pour le même client ; step ne relit pas le portefeuille | Conserver |
| États cachés | Trois CustomerTraits immuables, échantillonnés une fois ; creditworthiness conditionnelle au score | Ajouter stabilité du revenu et facteur commun corrélé |
| Revenu | Constant pendant tout l'épisode | Introduire évolution logarithmique bornée, réversion et choc adverse |
| Spending | Persistance, revenu, propension, headroom et choc lognormal | Conserver les canaux ; individualiser l'élasticité de limite |
| Paiement | Fraction persistante, logit comportemental, défaut de paiement, cap de capacité lié au revenu | Conserver, utiliser revenu actualisé et intensité macro |
| Comptabilité | B'=B-P+C ; pas de capitalisation ; dette au-dessus de la limite conservée | Conserver et expliciter la convention |
| Delinquency | Nombre de mois consécutifs de paiement insuffisant, cure possible, historique six mois | Conserver ; ajouter buckets diagnostiques 0/1/2/3+ sans prétendre à un vrai DPD |
| Défaut | Logit indépendant du PD model, exposition finale, événement terminal | Extraire dans un module dédié ; ajouter effet de baisse de revenu observée |
| Macro | Deux états, Markov dans la transition client ; RNG séparé mais pas de chemin sérialisable | Trois régimes, trois facteurs, chemins explicites et quatre scénarios |
| Hasard | Nombre fixe de draws par step ; default et behavior partagent le Generator | Canaux nommés pré-générés, customer/month/seed ; défaut indépendant |
| Information agent | Allowlist des features ; pas de latent ou vraie PD dans obs/info/history | Conserver ; historique scientifique privilégié distinct et opt-in |
| PD | Proxy logistique et GB de snapshots ; mismatch reconnu | Conserver ; retirer Pandas de l'inférence online du GB |
| Reward | Perte réalisée une fois ; intérêts/frais/funding/capital/pénalité | Conserver les coefficients et la définition |
| Validation | Diagnostics sur six clients et stress d'un seul type client | Passer à 5 000 clients, interventions appariées, check d'apprenabilité |

Les principaux coefficients existants étaient tous **supposés**, pas calibrés :
intercept du hazard -6, charges utilisation 1.2, dette/revenu 0.45, delinquency 0.65,
creditworthiness -0.8 ; persistance paiement 0.5, spending 0.35 ; élasticité commune
0.15. Les paramètres nouveaux restent synthétiques et sont explicités avec unités.
Aucun coefficient de reward n'a été choisi sur la performance d'une policy.

Canaux causaux déjà corrects : limite -> utilisation mécanique -> paiement/risque,
limite -> headroom/demande -> achats -> balance/exposition, puis effets sur les mois
suivants. Faiblesses : revenu fixe, faible corrélation des traits, macro individuelle
non partageable, aucun diagnostic exporté du hazard, validation populationnelle absente.

Pas de fuite directe trouvée dans les observations. L'adaptateur GB calculait toutefois
un DataFrame à chaque prédiction et les historiques ne permettaient pas d'auditer le
hazard caché. Le Sprint 2 traite ces points sans prétendre résoudre la calibration PD.
Les résultats du Sprint 1 restent ceux de la version 1 ; les sorties v2 utilisent de
nouveaux noms/répertoires et une version de DGP dans leur manifeste.
