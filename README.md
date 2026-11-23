# FRONTIERE_EFFICIENTE — Portfolio Construction Pipeline

Application Streamlit de construction de portefeuille utilisée pour le **point de gestion hebdomadaire**
(équipe Multi-Asset / ESR, Amundi). À partir d'historiques de prix exportés de Bloomberg, elle trace la
frontière efficiente et donne, pour chaque cible de volatilité, le panier qui **maximise le rendement**
sous contraintes de poids (par titre, par secteur GICS ou par classe d'actifs).

Tout le code est dans `pipe.py` (un seul fichier, ~1 600 lignes).

---

## 1. Lancer

```bash
pip install streamlit pandas numpy scipy matplotlib openpyxl
streamlit run pipe.py
```

Il n'y a pas encore de `requirements.txt` : la ligne ci-dessus liste les paquets importés par `pipe.py`.

Deux onglets : **📊 Equity** et **🌍 Multi-Asset**.

---

## 2. Onglet Equity

### Données

Par défaut l'app charge le classeur maître :

```
P:\GESTION_RPSE\GES - Gestion Epargne Salariale\16_PERSONNEL\Nadir\FRONTIERE_EFFICIENTE\UNIVERS_BLOOM.xlsx
```

(constante `EQ_DEFAULT_FILE_PATH` en haut de la section Equity ; on peut aussi uploader un autre fichier).
L'app affiche la date de dernière modification du fichier pour vérifier qu'il a bien été rafraîchi.

Le classeur est rafraîchi **chaque jour** par une macro VBA lancée par le Planificateur de tâches Windows
sur un poste Bloomberg (formules BDH / BDS), puis enregistré : Python lit les valeurs enregistrées.

**Onglets univers** (un par univers, on peut en combiner plusieurs) — format fixe :

| Ligne | Contenu |
|---|---|
| 3 | GICS 1 (secteur) |
| 4 | GICS 2 (industry group) |
| 5 | Ticker |
| 6-7 | ignorées |
| 8+ | prix ; dates en colonne A, un titre par colonne à partir de B |

**Onglet `BENCH`** : nom des indices en ligne 4 (col. B+, ex. `SX5E Index`), prix à partir de la ligne 7,
dates en colonne A. Plusieurs indices possibles, choix dans un menu déroulant.

**Onglet `BENCH_POIDS`** : poids du benchmark par secteur GICS 1 (secteur en col. A, poids en col. B+).
**Onglet `GICS 2`** (ou `GICS_2`, `GICS2`) : même chose par industry group GICS 2.

### Paramètres

- taux sans risque annuel, fenêtre historique (années, 5 par défaut) ;
- rendements **absolus** ou **excédentaires** vs l'indice choisi (R(titre) − R(bench)) ;
- plage de cibles de volatilité et nombre de cibles (9 par défaut entre 1 % et 25 %) ;
- poids max par défaut par titre, nombre de portefeuilles aléatoires pour le nuage, nombre de redémarrages de l'optimiseur.

### Contraintes

- **par titre** : tableau éditable (Min % / Max %) ;
- **par secteur GICS 1 et GICS 2** : bornes préremplies à poids du benchmark ± x % (x réglable), modifiables ensuite.

---

## 3. Onglet Multi-Asset

Upload d'un fichier (pas de fichier par défaut), format fixe :

| Ligne | Contenu |
|---|---|
| 1 | type d'actif (classe) |
| 2 | nom |
| 3 | ISIN |
| 4+ | prix ; dates en colonne B, instruments à partir de la colonne C |

Classes : Equity, Fixed Income, Commodities, FX, Real Estate, Cash (synonymes reconnus). Un instrument sans
classe lisible passe en `Unclassified` et se reclasse à la main dans le tableau ; **l'optimisation est bloquée
tant qu'un instrument n'est pas classé** (choix volontaire, pour éviter un résultat faux sans alerte).

Contraintes par instrument (tableau) et par classe d'actifs (curseurs Min / Max).

**Configurations de fonds** : on peut enregistrer sous un nom les classifications et les bornes d'un fonds,
puis les recharger. Elles sont stockées dans `fund_configs.json`, à côté de `pipe.py` (Multi-Asset uniquement).

---

## 4. Méthode

1. Prix mis en **hebdomadaire** (dernier cours de la semaine), coupés à la fenêtre choisie ; les dates avec plus
   de 50 % de trous sont retirées, les titres avec moins de 80 % d'historique sont exclus.
2. Rendements hebdomadaires → rendement moyen et matrice de covariance annualisés (52 semaines).
3. Pour chaque cible de volatilité : **max rendement** sous `somme des poids = 1`, `vol ≤ cible`, bornes et
   contraintes de groupe. Optimiseur SLSQP de `scipy`, avec gradients analytiques et plusieurs points de départ
   aléatoires. Une cible sous la vol minimale atteignable renvoie le portefeuille de variance minimale, une cible
   au-dessus de la vol max renvoie le portefeuille de rendement max (signalés dans le tableau).
4. Graphique : nuage Monte Carlo de portefeuilles aléatoires, points de la frontière colorés par Sharpe,
   barres de Sharpe par cible.

### Sorties

Tableau des résultats (rendement, vol, Sharpe par cible), poids des paniers par cible,
téléchargements `efficient_frontier_baskets.csv`, `efficient_frontier.png`, `prices.csv`, `returns.csv`,
`bench_returns.csv`, `excess_returns.csv`.

---

## 5. Points d'attention

- Le format des fichiers est lu **par position de ligne**, pas par libellé : insérer une ligne dans le classeur
  Bloomberg décale tout. Si l'export change, modifier les constantes `EQ_ROW_*`, `BENCH_ROW_*`, `MA_ROW_*`.
- Rendements et covariances sont historiques (pas de prévision) : la frontière décrit le passé de la fenêtre choisie.
- Le chemin du fichier par défaut est sur le P: : à changer si le dossier est déplacé.
