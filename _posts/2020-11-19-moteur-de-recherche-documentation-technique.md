---
layout: post
title:  "Moteur de recherche pour documentation technique"
date:   2020-11-19 18:30:00 +0200
tags: FR NLP
---

Nous sommes actuellement 4 étudiants en école d’ingénieur (Brice Chichereau[^brice], Artur Vianna[^artur], Raphaël Breteau[^raphael] et Alide Xiang[^alide]) réalisant leur année de césure à la Digital Tech Year[^DTY]. D’une durée de 6 mois, le programme a pour but de former les étudiants au travers de trois projets de sept semaines pour diverses entreprises françaises dans le but de fournir des prototypes fonctionnels répondant à une problématique donnée. Les projets sont orientés autour du développement web et mobile et/ou de l’intelligence artificielle.

Chaque prototypage est également l’occasion d’appliquer une méthodologie de projet Agile, notamment via la division en sprints de une semaine du calendrier et l’organisation de nombreuses réunions avec le client pour discuter des avancées passées et futures du projet. Pour réaliser ces projets, nous avons été encadrés tout au long de ces 7 sept semaines par un coach technique spécialisé dans le domaine du Question Answering venant l’entreprise Illuin. Il nous a ainsi apporté son expertise pour nous orienter vers les solutions les plus efficaces pour répondre aux attentes du client.

Notre premier projet pour la DTY[^DTY] nous a amené à travailler pour Enedis[^Enedis], distributeur d’électricité, en collaboration avec une de leurs équipes de data scientists. Enedis[^Enedis] dispose en effet d’une documentation technique conséquente dans laquelle la moindre recherche d’information de la part d’un employé (technicien, ingénieur ou membre de la R&D) peut s’avérer complexe du fait de la grande quantité de documents disponibles et de leur taille.

La recherche à la main d’informations dans le corpus pose donc 2 problèmes majeurs :

* Une perte de temps due à la recherche
* Une perte d’information due à la taille du corpus

Ainsi, la problématique du projet était d’explorer et de développer une solution permettant une récupération automatique de l’information pertinente dans le corpus de documents qui répond à une question posée par un employé d'Enedis[^Enedis].

### Méthode

La situation de base dans laquelle nous étions était la suivante : nous avions accès à une liste de documents techniques découpés au préalable par les équipes d'Enedis[^Enedis] en fichiers `.txt` contenant chacun plus ou moins un paragraphe entier. À partir de ces fichiers textes (que nous appellerons chunks dans la suite) il nous fallait, pour une question d’un utilisateur, trouver une réponse pertinente.

Avant de commencer à écrire la moindre ligne de code, il nous fallait nous familiariser avec l’état de l’art concernant notre problème. Nous avons donc dédié une semaine entière à la lecture d’articles de recherche à propos de différentes méthodes de NLP qui pourraient répondre à notre problème.
De cette recherche documentaire, nous avons constaté que les systèmes de Question Answering étaient souvent composés de 2 parties principales :

* Un algorithme d’Information Retrieval récupérant les contextes les plus pertinents à la question.
* L’algorithme de Question Answering en lui même (souvent basé sur un modèle de type Transformers).

Nous avons donc pris la décision d’adopter une architecture similaire pour notre solution, et pour chacune des deux parties indépendantes, nous nous sommes rapidement rendus compte qu’il existait de très nombreuses méthodes différentes qu’il faudrait essayer et évaluer.

### Information Retrieval

La première partie de notre solution consiste en un algorithme de Information Retrieval qui a pour objectif de récupérer, pour une question donnée, les chunks les plus pertinents vis à vis de cette question.
Avant toute chose il nous fallait définir une métrique d’évaluation des différents algorithmes que nous allions tester. Nous avons décidé d’adopter la métrique suivante :

![accuracy]({{site.baseurl}}/assets/img/enedis_ir_accuracy.png)
{: style="text-align: center;"}

Avec N le nombre d’éléments dans notre dataset de test.
Cette métrique représente en quelque sorte le taux de réussite de notre algorithme pour un nombre de chunks pertinents donné.

#### TfidfVectorizer

La première méthode d’Information Retrieval que nous avons testé est basée sur un calcul de similarité entre embeddings (ie représentation vectorielle d’un texte) générés par un algorithme basé sur **TF-IDF**[^TF-IDF].
La méthode pour construire les embeddings est la suivante :

1. Génération d’un vocabulaire de taille fixe n en parcourant les documents.
2. Pour chaque document on crée un vecteur de taille n avec à l’indice i : tf_i * idf_i
    1. tf_i = fréquence du mot i dans le document
    2. idf_i = terme de pondération décroissant en le nombre de documents où apparaît le mot i

On construit ainsi un embedding pour chacun des chunks ainsi qu’un embedding pour la question posée par l’utilisateur (on peut stocker les embeddings des chunks dans un fichier numpy pour éviter les calculs redondants).

Une fois ces embeddings construits on calcule pour chaque embedding de chunk sa **similarité cosinus**[^SimCosinus] avec l’embedding de la question. On peut ensuite trier les chunks par ordre décroissant de similarité afin de récupérer les chunks les plus pertinents.

Notre implémentation utilise la classe TfidfVectorizer du module Python scikit-learn[^TfidfVectorizer]. On obtient avec cet algorithme et différentes tailles de vocabulaire les résultats suivants :

![TF-IDF]({{site.baseurl}}/assets/img/enedis_tfidf.png)
{: style="text-align: center;"}

Figure 1. Performance de TF-IDF en fonction du nombre de chunks récupérés
{: style="font-style: italic; text-align: center;"}
&nbsp;

#### Whoosh BM25F

Après ces premiers résultats assez encourageants, nous nous sommes tournés vers un autre algorithme d’Information Retrieval qui est revenu souvent dans nos recherches : **BM25**. Cet algorithme permet de calculer, avec une méthode fréquentielle, un score de similitude entre deux documents.

Nous avons plus particulièrement utilisé le module Python Whoosh[^Whoosh] qui implémente une génération d’indices pour les documents et un système de recherche utilisant le score BM25F. Avec cette méthode-ci nous avons obtenu les résultats suivants :

![BM25]({{site.baseurl}}/assets/img/enedis_bm25.png)
{: style="text-align: center;"}

Figure 2. Performance de BM25F et TF-IDF en fonction du nombre de chunks récupérés
{: style="font-style: italic; text-align: center;"}
&nbsp;

On remarque que la méthode BM25 est en général supérieure à TF-IDF, c’était donc un pas dans la bonne direction.

Plus tard, en explorant les questions non répondues par Whoosh[¨Whoosh], nous avons remarqué qu’il y avait plusieurs questions dans lesquelles il y avait des variations (pluriels, accents manquants) de certains mots par rapport à ceux de la réponse présente dans le chunk. Nous avons alors traité ces cas en ajoutant les Analyser présents dans le module Whoosh[^Whoosh] lors de l’indexing ce qui a amélioré les performances de façon notable :

![BM25]({{site.baseurl}}/assets/img/enedis_bm25.png)
{: style="text-align: center;"}

Figure 3. Evolution des performances avec les variations de BM25
{: style="font-style: italic; text-align: center;"}
&nbsp;

#### Doc2Vec

En parallèle des travaux sur les méthodes fréquentielles, nous avons également exploré des méthodes vectorielles, le coeur de cette approche est de savoir comment représenter le texte à l’aide de vecteurs afin de calculer une distance entre les représentations vectorielles des chunks et des questions pour procéder à la tâche d’Information Retrieval.

La première piste a été d’utiliser un modèle **Word2Vec**[^Word2Vec] français déjà entraîné contenu dans le module Python **fastText** pour vectoriser les mots de chaque chunk et question puis faire la moyenne de ces vecteurs afin d'avoir une vecteur par paragraphe. Cette approche s’est avérée peu concluante aboutissant à des résultats quasi nuls.

La deuxième piste a été d’entraîner un modèle **Doc2Vec**[^Doc2Vec] proposé dans le module gensim, cet entraînement est basé sur le MLM (Masked Language Modeling) consistant à masquer un mot dans un contexte et le modèle essaie de trouver ce mot. Cet apprentissage peut se faire de deux façons, soit avec la méthode DM (Distributed-Memory) ou bien DBoW (Distributed bag of words), la grosse différence est que la méthode DBoW ne prend pas en considération le contexte dans la phase MLM alors que DM le fait[^DMvsDBoW]. Plusieurs essais d’entraînement ont été faits en faisant varier la taille du vecteur et la méthode d’entraînement. Dans notre cas, la façon d’entraîner a influencé grandement les performances finales faisant passer les performance de 14% (DM) à 54% (DBoW) pour les 5 premiers chunks.

Dans la globalité les résultats sont moins bon que les méthodes fréquentielle :

![doc2vec]({{site.baseurl}}/assets/img/enedis_doc2vec.png)
{: style="text-align: center;"}

Figure 4. Performance de Doc2Vec en fonction de la taille des vecteurs
{: style="font-style: italic; text-align: center;"}
&nbsp;

Une troisième piste a été de voir s’il était pertinent de combiner cette méthode aux méthodes fréquentielles (je vais les résumer à BM25 pour la suite), tout d’abord il était intéressant de voir si **Doc2Vec**[^Doc2Vec] performait sur des questions supplémentaires à BM25 et c’était le cas sur notre jeu de test. Nous avons alors combiné les deux méthodes en gardant un nombre **range** de chunks en sortie de BM25 et Doc2vec puis on normalise les scores BM25 qu’on pondère avec les distances de Doc2Vec pour chaque chunk pour finalement avoir en sortie le nombre de chunks pertinents. Cette méthode a abouti aux résultats suivants sur le jeu de test d’Enedis :

![doc2vec-bm25]({{site.baseurl}}/assets/img/enedis_doc2vec_bm25.png)
{: style="text-align: center;"}

Figure 5. Performance de Doc2Vec+BM25 en fonction de la taille des vecteurs
{: style="font-style: italic; text-align: center;"}
&nbsp;

D’autres essais ont été effectués notamment sur un jeu de test qui a été annoté par notre équipe dont la caractéristique principale est l’effort de reformulation des mots contenus dans la réponse pour façonner la question et en faisant varier la **range** :

![doc2vec-enedis]({{site.baseurl}}/assets/img/enedis_doc2vec_enedis.png)
{: style="text-align: center;"}

Figure 6. Performance sur le jeu de test ENEDIS
{: style="font-style: italic; text-align: center;"}

![doc2vec-dty]({{site.baseurl}}/assets/img/enedis_doc2vec_dty.png)
{: style="text-align: center;"}

Figure 7. Performance sur le jeu de test DTY
{: style="font-style: italic; text-align: center;"}
&nbsp;

De ces essais, on en tire plusieurs résultats intéressants, tout d’abord les performances de BM25 sont plus faibles sur le second jeu de test ce qui est explicable par le fait de la reformulation qui détériore les approches fréquentielles. De plus, les performance de la combinaison des méthodes BM25 et Doc2vec fonctionne mieux sur le deuxième jeu de données, cela s’explique par le fait que les modèles de embeddings traitent mieux les synonymes caractéristiques de ce jeu de test. Enfin, on constate que les performances lié à la valeur de la range varient grandement du jeu de test, d’un côté pour range=2*nb_chunk les résultats sont améliorés dans le second tableau alors que c’est la tendance inverse pour le premier.

#### DPR

La dernière approche que nous avons tenté pour la tâche d’Information Retrieval repose sur le modèle DPR ou Dense Passage Retrieval. Issu d’un article de recherche publié en mai 2020 par Karpukhin, Vladimir, et al,[^DPR] ce modèle est lui-même basé sur le modèle Transformers BERT de Google. En effet, le but du système est d’obtenir un encodage sous forme de vecteurs de la question et des contextes pour pouvoir ensuite en calculer une mesure de similarité via un produit scalaire. Le modèle repose sur deux réseaux de type BERT entraînés en parallèle via une fonction de coût et un jeu de données d’entraînement particulier.

![DPR]({{site.baseurl}}/assets/img/enedis_dpr.png)
{: style="text-align: center;"}

Figure 8. Représentation de l’organisation globale de DPR
{: style="font-style: italic; text-align: center;"}
&nbsp;

Ainsi, chaque paragraphe est encodé par un des réseaux et la question par l’autre réseau. On calcule ensuite la distance entre ces encodages et enfin, via un classement des métriques obtenues, on récupère les meilleurs chunks répondant à la question.

Pour utiliser ce modèle, deux approches coexistent:

1. Utiliser la version de DPR déjà présente sur Huggingface mais uniquement entraînée sur du texte en anglais
2. Réécrire un modèle DPR depuis le début puis l’entraîner sur un jeu de données en français

Nous avons ainsi testé les deux approches en commençant par celle utilisant la version DPR disponible dans la librairie Transformers.

**DPR HuggingFace**

Il existe un modèle DPR disponible sur la librairie Huggingface qui est un modèle pré-entrainé sur un dataset anglais. Pour l’exploiter, nous avons traduit nos chunks en anglais avec le module googletrans[^googletrans], ensuite nous découpons les chunks en sous-chunks de façon à obtenir des sous-chunks de 512 tokens exploitables par le modèle. Puis nous faisons la moyenne des vecteurs pour représenter le chunk afin de procéder à la tâche d’information retrieval. Cette méthode a été dans l’ensemble rapide à implémenter sans compter l’algorithme de découpage de chunks. Elle a abouti à des résultats de l’ordre des 30% pour 5 chunks récupérés, ce sont des résultats moins convaincants que doc2vec. Cela est dû sans doute à la perte d’information lors de la traduction et le découpage des chunks.

**DPR "from scratch"**

La deuxième implémentation de DPR partait quasiment de 0. En effet, comme il n’existait aucun modèle DPR disponible en français, nous avons décidé de créer le nôtre, étant donné les performances théoriques et pratiques d’un tel modèle. Pour le réaliser, nous nous sommes inspirés de la version de DPR disponible sur Huggingface en l’adaptant à une utilisation en français. Pour cela, les modèles BERT en anglais des encodeurs de contexte et de question ont été remplacés par des modèles CamemBERT entrainés en français.

Il a également été nécessaire de créer un jeu de données d’entraînement à partir de FQuAD[^FQuAD] 1.0, le dataset de Question Answering en open source créé par Illuin. L’entraînement de DPR se fait grâce à des échantillons divisés en 3 parties:

* Une question provenant de FQuAD
* Un contexte positif à cette question, c’est-à-dire contenant la réponse
* Un nombre k>1 de contextes négatifs, ne contenant pas la réponse

On entraîne ensuite le modèle grâce à une fonction de coût particulière, inspirée de la Negative Log-Likelihood (ou NLL) loss :

![DPR Loss]({{site.baseurl}}/assets/img/enedis_dpr_loss.png)
{: style="text-align: center;"}

Figure 9. Fonction de coût de l'entraînement de DPR
{: style="font-style: italic; text-align: center;"}
&nbsp;

Où q_i désigne la question i, p_i^+ le contexte positif pour cette question et p_i^- les n contextes négatifs associés à cette même question. sim(i,j) désigne finalement la similarité entre i et j, obtenue soit via la norme L2 de la différence des vecteurs i et j, soit via la similarité cosinus. Dans les faits, nous avons utilisés quasiment exclusivement la similarité L2.

Ce modèle a ensuite été entraîné sur le cluster de GPUs de CentraleSupélec. En effet, au vu de la taille du jeu de données d’entraînement (plus de 20.000 questions disponibles dans FQuAD 1.0) et du temps d’exécution pour un échantillon sur CPU (environ 10-15 secondes), le choix le plus raisonnable était d’utiliser les GPUs de CentraleSupélec. Après avoir adapté le code, nous avons entraîné le modèle sur plusieurs epochs (entre 4 et 6) pour une durée d’entraînement d’environ 1h20 par epoch (en comptant la validation effectuée à chaque epoch).

Malheureusement, les résultats en sortie du modèle se sont révélés très inférieurs à nos attentes concernant les prévisions. Nous avons donc eu plusieurs idées pour améliorer ces performances, notamment en vérifiant que la backpropagation de notre fonction de coût modifiait bien tous les poids ou bien même en modifiant notre fonction de coût pour y intégrer la NLL Loss de Torch, comme dans la version de HuggingFace. La fin du projet approchant, nous n’avons pas eu le temps d’implémenter ces solutions et de les tester, et il conviendra donc aux Data Scientists de les essayer s’ils souhaitent intégrer ce modèle dans le pipeline d’Information Retrieval.

### Question Answering

#### Objectif

L’objectif classique du Question Answering est de trouver dans un texte (souvent appelé aussi *contexte*) le passage qui répond à une question donnée. Dans notre cas, il y a plusieurs contextes, les chunks pertinents qui ont été sélectionnés par l’algorithme d’IR.

Nous détaillerons dans la conclusion de l’article comment l’intégration de l’IR et du QA est faite, donc dans cette section nous considérons que nous avons seulement un chunk en entrée et qu’il contient forcément le passage qui répond à la question.

Si notre algorithme de Question Answering était une boîte noire, il se présenterait comme ça :

![QA Boîte Noire]({{site.baseurl}}/assets/img/enedis_qa_boite_noire.png)
{: style="text-align: center;"}

Figure 10. Schéma global de l'algorithme de Question Answering
{: style="font-style: italic; text-align: center;"}
&nbsp;

#### Modèle

Après une recherche dans la littérature pour les modèles d’état de l’art dans le domaine, nous avons décidé d’utiliser un modèle Transformer déjà entraîné pour comprendre la langue française.

Plus précisément, nous avons pris un modèle du type CamemBERT[^CamemBERT] entraîné sur la tâche de Question Answering dans le dataset français FQuAD[^FQuAD]. Ce modèle a été développé par Illuin Technologies[^Illuin] et il est disponible sur des versions *base* et *large* à travers la librairie HuggingFace[^HuggingFace].

Pour commencer, nous avons utilisé le modèle sans aucun changement pour ensuite analyser les résultats et trouver des points d’amélioration.

De façon plus précise, le modèle nous donne deux scores pour chaque token du contexte qui indiquent le niveau de confiance il soit le premier ou le dernier token de la réponse (*start_score* et *end_score* respectivement). Avec ceci, on prend la position du token qui a le *start_score* le plus élevé comme position de début de notre réponse et on découpe le contexte à partir de cette position jusqu’à la position du token qui a le *end_score* le plus élevé. En plus on peut faire une moyenne entre les deux scores pour obtenir un score de confiance globale du modèle pour une réponse donnée.

#### Solution Préliminaire

Le problème avec cette stratégie est que tous les modèles basés sur BERT cherchent à trouver le passage du texte le plus court qui répond à la question et souvent, vu le domaine très technique des documents, la réponse attendue est plus longue.

Donc, après avoir le petit passage sorti du modèle, nous allons retenir comme réponse finale la phrase du contexte qui contient ce passage.

![QA Architecture 1]({{site.baseurl}}/assets/img/enedis_qa_archi1.png)
{: style="text-align: center;"}

Figure 11. Première architecture de l'algorithme de Question Answering
{: style="font-style: italic; text-align: center;"}
&nbsp;

#### Métrique d'évaluation

Pour évaluer la performance de notre algorithme nous avons tout simplement utilisé un scoring F1 (la même métrique utilisée par Stanford pour évaluer SQuAD). D’abord, nous tokenisons notre réponse avec le tokenizer de CamemBERT et la réponse correcte. Ensuite nous calculons, le score F1 sur l’ensemble des tokens.

Le résultat est un pourcentage qui indique la similarité entre notre réponse et celle du test set (plus précisément entre les tokens de chaque réponse).

#### Résultats préliminaires

Avec la métrique établie précédemment, nous obtenons l’histogramme suivant :

![QA Résultats 1]({{site.baseurl}}/assets/img/enedis_qa_res1.png)
{: style="text-align: center;"}

Figure 12. Répartition des scores F1 pour le dataset de test Enedis
{: style="font-style: italic; text-align: center;"}
&nbsp;

Si on établit que pour être considérées comme correctes nos réponses doivent avoir un score minimal de 0.6 (doivent être au moins 60% similaires à la vraie réponse), on trouve que dans 43.8% des cas nous trouvons la bonne réponse.

Ces résultats n’étant pas du tout satisfaisant, nous sommes partis en recherche d'améliorations.

#### Améliorations - tokens non reconnus par le CamemBERT

La première chose que nous avons remarqué dans les résultats de nos tests est la présence des `<unk>` dans beaucoup de réponses. En fait, c’est comme ça que le tokenizer de CamemBERT représente les tokens qu’il ne reconnaît pas. Donc, nous avons activé une option du tokenizer de HuggingFace pour enlever ces tokens.

Par contre, faire cela pose un autre problème parce que le texte correspondant aux tokens non reconnus est toujours présent dans le chunk et on veut toujours récupérer l’intégralité de la phrase du chunk qui contient le passage. Une correspondance exacte n’étant pas possible à cause de cette différence entre le passage donné par le modèle et le chunk. Donc, pour trouver la phrase qui contient le passage nous calculons le score de similarité F1 entre toutes les phrases du contexte et nous retenons la phrase qui a le score le plus élevé. En détail :

1. Obtention du passage qui répond à la question sans les tokens non reconnus.
2. Découpage du contexte entier (avec tous ces tokens) en phrases.
3. Calcul de la métrique F1 entre le passage et toutes les phrases.
4. La phrase qui a le score le plus élevé est retenue comme contenant le passage.

#### Améliorations - chunks trop longs, algorithme de Découpage

Un autre souci est que l’input de notre modèle (question + chunk) était souvent trop long. Les modèles basés sur BERT ont une taille maximale d’entrée de 512 tokens. Pour traiter cela, il fallait découper notre chunk en plusieurs parties et faire passer chaque partie par le modèle avec la question.

Certaines techniques pour utiliser les contextes de plus de 512 tokens avec un modèle type BERT existent, mais notre temps de développement était court. Donc, nous avons décidé de faire un algorithme simple qui découpe les chunks en regroupant ses phrases de façon à préserver la logique raisonnablement.

Il est important que chaque partie du chunk préserve au maximum la logique du contexte entier parce que le modèle va uniquement prendre en compte l’information d'une partie pour répondre à la question. Donc, si on découpe une phrase au milieu par exemple, on risque de ne pas passer au modèle tout le passage dont il a besoin pour répondre à la question.

Algorithme de découpage d’un chunk en N parties :

1. N = 2.
2. Découpage du chunk en phrases.
3. Regroupement de ces phrases en N groupes en préservant l’ordre.
4. Vérification que tous les groupes font moins de 512 tokens de longueur, si un groupe en fait plus :
    1. N += 1
    2. Retournez à l’étape 3

#### Améliorations - chunks trop longs, max_score()

À la sortie de notre algorithme on aura N sous-chunks qu’on fera passer par le modèle de QA chacun avec la question et cela nous donnera N passages du chunk qui peuvent contenir la réponse (chacune issue d’un sous-chunk différent). Heureusement, on peut utiliser le score de confiance que le modèle a donné à chaque réponse pour faire cette décision. On retiendra donc la réponse qui a le score le plus élevé.

#### Solution retenue

Finalement, voilà l’architecture globale de notre solution :

![QA Architecture Finale]({{site.baseurl}}/assets/img/enedis_qa_final.png)
{: style="text-align: center;"}

Figure 13. Architecture finale de l'algorithme de Question Answering
{: style="font-style: italic; text-align: center;"}
&nbsp;

On découpe le chunk avant et on fait passer chacun des sous chunk par le modèle. Ensuite on récupère les N passages des sous chunks donnés par le modèle et on retient celui auquel le modèle a donné le score le plus élevé. Finalement, on calcule la métrique de similarité F1 entre le passage retenu et toutes les phrases du chunk et on retient comme réponse celle qui est la plus similaire au passage.

#### Nouvelle métrique d’accuracy

En regardant les résultats de nos tests, nous avons constaté qu’il n'y avait presque jamais une correspondance exacte entre notre réponse et celle du test set. Parfois la nôtre contenait l’autre, parfois le contraire. Il y avait même des cas où les deux réponses faisaient à peu près la même taille et il y avait une intersection importante entre les deux.

Donc, au lieu d’utiliser un score de 0.6 établi presque arbitrairement, nous allons trouver dans le chunk les phrases qui correspondent au passage correct du test set (golden answer). Pour faire cela on découpe la golden answer en phrases (si besoin) et on prendra les phrases du contexte qui ont le score de similarité F1 le plus élevé avec celles-là. C’est exactement la même chose que l’on fait dans la dernière étape de notre solution.

Vu que notre réponse et les phrases de la golden answer sont sorties de l’ensemble de phrases du contexte, on peut juste les comparer sans inquiétude. Si notre réponse se trouve dans l’ensemble des phrases de la golden answer, on la considère comme correcte pour le calcul de l’accuracy, sinon on la considère comme fausse.

#### Résultats finaux

![QA Architecture Finale]({{site.baseurl}}/assets/img/enedis_qa_res.png)
{: style="text-align: center;"}

Figure 14. Résulats finaux de l'algorithme de Question Answering
{: style="font-style: italic; text-align: center;"}
&nbsp;

Avec une accuracy de 87.5 % on arrive à augmenter énormément nos performances !

Il est important de remarquer que le score utilisé pour construire l’histogramme sert juste comme un complément à l’accuracy. Il n’est plus utilisé pour la calculer. Maintenant, l’histogramme nous permet de vérifier que l’accuracy que nous avons trouvée est cohérente.

### Conclusion

La structure globale de notre prototype se compose d’une partie **Information Retrieval** qui nous renvoie les 5 chunks les plus pertinents (avec une performance d’environ 81%) sur lesquels nous allons effectuer la tâche de **Question Answering** (avec une performance de 80%) et qu’on classera à l’aide du score de confiance fourni par le modèle QA. Il faudra confronter le modèle à des situations réelles et à d'autres use-cases afin d'en valider les performances.

Si le projet était amené à être continué il y aurait plusieurs axes vers lesquels on peut s’orienter. D’une part finir le travail commencé sur DPR qui reste une méthode très prometteuse pour faire de l’Information Retrieval et ensuite faire un travail similaire à Doc2Vec de combinaison avec BM25 afin d’évaluer les différentes performances, d’autre part explorer des axes d’améliorations du système de QA comme par exemple un entraînement adversarial pour pouvoir comprendre les contextes ne contenant pas la réponse ou bien faire une exploration de fine-tuning des modèles CamemBERT sur les textes concernés afin d’améliorer la tâche de QA.

### Remerciements

Nous tenons à remercier Quentin Heinrich qui a su nous guider et nous accompagner tout au long du projet, ses insights furent primordiaux dans l'élaboration de la solution que nous avons atteinte.

Nous souhaitons aussi remercier les équipes d'Enedis avec qui nous avons travaillé pour leur grande implication dans le bon déroulé de ce projet.

Enfin nous voulons remercier Gwladys LEFEVRE qui nous a accompagné pour toute la partie business du projet pour ses précieux conseils.

### Liens externes

[^brice]: [Brice Chichereau](https://www.linkedin.com/in/brice-chichereau/)
[^raphael]: [Raphaël Breteau](https://www.linkedin.com/in/rapha%C3%ABl-breteau/)
[^artur]: [Artur Vianna](https://www.linkedin.com/in/artur-vianna-9a987717a/)
[^alide]: [Alide Xiang](https://www.linkedin.com/in/alide-xiang/)
[^DMvsDBoW]: [A simple explanation of document embeddings generated using Doc2Vec](https://medium.com/@amarbudhiraja/understanding-document-embeddings-of-doc2vec-bfe7237a26da)
[^DTY]: [Digital Tech Year](https://paris-digital-lab.com/digital-tech-year-fr/)
[^Enedis]: [Enedis](https://www.enedis.fr/)
[^TF-IDF]: [TF-IDF](https://fr.wikipedia.org/wiki/TF-IDF)
[^SimCosinus]: [Similarité Cosinus](https://fr.wikipedia.org/wiki/Similarit%C3%A9_cosinus)
[^TfidfVectorizer]: [Documentation de la class TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
[^Word2Vec]: [Word2Vec](https://en.wikipedia.org/wiki/Word2vec)
[^fastText]: [fastText](https://fasttext.cc/)
[^Doc2Vec]: [Doc2Vec](https://radimrehurek.com/gensim/auto_examples/tutorials/run_doc2vec_lee.html)
[^DPR]: [Karpukhin, Vladimir, et al. "Dense Passage Retrieval for Open-Domain Question Answering." arXiv preprint arXiv:2004.04906 (2020).](https://arxiv.org/abs/2004.04906)
[^CamemBERT]: [CamemBERT](https://camembert-model.fr/)
[^FQuAD]: [FQuAD](https://fquad.illuin.tech/)
[^Illuin]: [Illuin Technologies](https://www.illuin.tech/)
[^HuggingFace]: [HuggingFace](https://huggingface.co/)
[^Whoosh]: [Whoosh](https://whoosh.readthedocs.io/en/latest/intro.html)
[^googletrans]: [googletrans](https://pypi.org/project/googletrans/)
