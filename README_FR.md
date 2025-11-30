[中文版本](README.md)，👉Version Française

# **Résumé d’avancement du projet Whisper-Hailo8L**

### **Objectif du projet**

Mon objectif à long terme est de permettre à Whisper de fonctionner en temps réel sur une plateforme composée d’un Raspberry Pi 5 et d’un accélérateur Hailo-8L. Pour atteindre ce résultat, il ne s’agit pas simplement de quantifier le modèle original : je dois reconstruire un encodeur entièrement compatible avec les contraintes matérielles du NPU, puis transférer les connaissances du modèle maître vers ce nouvel encodeur au moyen d’une distillation.
 Autrement dit, je ne fais pas simplement du déploiement de Whisper, mais je conçois une version « Whisper-Lite » adaptée à Hailo.

### **Conception de l’encodeur et du décodeur**

**Encodeur**

Dans la première version, j’ai conçu un encodeur reposant sur une attention linéaire : l’idée était de remplacer l’attention classique par une kernel attention afin d’éviter softmax et les multiplications matricielles volumineuses, difficiles à gérer sur Hailo. Les deux premières couches convolutionnelles ont également été modifiées, et la sortie temporelle a été fixée à 500 pas au lieu des 1500 de Whisper-small.

Même si cette structure fonctionnait correctement en PyTorch, elle a introduit un problème majeur lors de la distillation. En modifiant la longueur du contexte, les positions temporelles du modèle enseignant ne correspondaient plus à celles de mon modèle. La distillation força donc l’étudiant à imiter des représentations latentes déjà tronquées. Le processus finissait par converger, mais l’encodeur appris ne reflétait plus fidèlement les représentations de Whisper.

Lorsque j’ai tenté de porter cet encodeur sur Hailo, les difficultés se sont amplifiées. Le compilateur de Hailo modifie activement le graphe computationnel : LayerNorm devient GroupNorm accompagné de reshapes, les étapes de normalisation dans la kernel attention sont réorganisées, certains paddings sont supprimés et des reshapes fusionnés ou éliminés. Après quantification, le graphe résultant n’était plus du tout celui que j’avais entraîné en PyTorch, et les résultats d’inférence divergeaient fortement.
 Lors de certaines tentatives, la phase de quantification consommait même plusieurs centaines de gigaoctets de mémoire, ce qui montre clairement que cette architecture n’est pas compatible avec l’optimisation interne du compilateur Hailo.

**Décodeur**

La partie décodeur a posé des problèmes encore plus importants. J’ai d’abord essayé un décodeur basé sur le CTC, mais celui-ci est incapable d’apprendre le contexte linguistique et a tendance à produire beaucoup de tokens « blancs ». Cela contredit complètement le rôle du décodeur dans Whisper, qui agit comme un véritable modèle de langage.

J’ai ensuite expérimenté des approches inspirées de Mamba et des SSM. L’espoir était d’utiliser leurs propriétés convolutionnelles et récurrentes pour éviter les calculs d’attention auto-régresseurs. Cependant, la théorie de Mamba repose sur un noyau de convolution de longueur potentiellement infinie. Sur Hailo, je suis obligé de fixer une longueur artificielle pour ce noyau : le modèle perd alors sa nature récursive et se transforme en un simple bloc convolutionnel de taille figée. Mathématiquement, cela dénature complètement le mécanisme, et pendant la quantification, le comportement du modèle s’éloigne fortement de celui observé durant l’entraînement.

Les choses deviennent encore plus complexes au niveau de la cross-attention. Même en remplaçant softmax par un noyau linéaire, les produits QK nécessitent une correspondance stricte des formes. Dès qu’un padding ou un broadcast est impliqué, le compilateur Hailo réécrit automatiquement la structure du graphe. À partir de ce moment-là, le modèle obtenu ne peut plus rester cohérent avec l’original.
 Cela signifie qu’un décodeur basé sur l’attention est, de fait, impossible à implémenter correctement sur Hailo.

**Décision : laisser le décodeur sur CPU**

Face à ces constats, j’ai compris que le décodeur n’est pas adapté à une exécution sur NPU. Il doit rester sur CPU (ou GPU), tandis que Hailo ne doit gérer que l’encodeur. Cette séparation correspond d’ailleurs naturellement à la philosophie de Whisper : l’encodeur extrait les caractéristiques acoustiques, alors que le décodeur effectue un travail de modélisation linguistique.
 Laisser la partie linguistique au CPU ne pénalise pas le temps réel.

### **Direction future**

La prochaine étape consiste donc à revenir au cœur du problème : la reconstruction complète de l’encodeur. Ce nouvel encodeur devra respecter strictement les contraintes du compilateur Hailo : même longueur temporelle que Whisper-small (idéalement 1500), formes entièrement déterministes, et aucune opération susceptible d’être réécrite par le compilateur.

Cela implique que je dois m’appuyer davantage sur des convolutions classiques, des DepthwiseConv, du GroupNorm et des opérations élémentaires, tout en évitant les reshapes complexes, les broadcasts dynamiques et les sommes irrégulières. Cette approche se rapproche davantage de la philosophie des architectures type YOLO, optimisées pour les NPUs embarqués.

Une fois que l’encodeur aura été redéfini, je suivrai la même procédure qu’auparavant : export en ONNX, conversion vers HAR, puis compilation en HEF. Si cette chaîne passe sans modifications non désirées, je pourrai procéder à la distillation depuis Whisper. Le résultat sera ensuite réutilisé directement avec le décodeur de Whisper.

Du point de vue purement technique, j’ai déjà mis en place une pipeline de génération de données très fiable. Grâce à un service gRPC, je peux récupérer de manière stable les sorties de l’encodeur sur Hailo (vecteurs latents), et j’ai constitué un corpus de 50 000 exemples destinés à l’apprentissage du décodeur. Cette partie est fonctionnelle et robuste.
 Les problèmes actuels se concentrent donc sur deux points essentiels : la structure interne de l’encodeur et la nécessité, lors de la distillation, de conserver une correspondance parfaite avec le contexte temporel de Whisper.

À l’avenir, mon travail consistera à construire un encodeur mieux adapté au hardware, dont la structure restera intacte après compilation, qui pourra recevoir correctement la distillation du modèle maître et qui permettra enfin la réalisation d’un véritable « Whisper embarqué » sur Hailo-8L.

(Note : le flux complet du compilateur Hailo repose sur trois étapes : conversion, quantification et compilation. La quantification peut déjà poser problème, car le compilateur génère parfois des opérations inutiles. Lorsque j’avais tenté de compiler un ancien encodeur, j’avais dû contourner la détection de bruit pour que le modèle passe la quantification. Par la suite, la compilation du fichier decoder.har s’était soldée par un échec, ce qui est cohérent avec la complexité extrême du graphe du décodeur. Je mets donc cette partie en pause afin de me concentrer entièrement sur l’encodeur.)