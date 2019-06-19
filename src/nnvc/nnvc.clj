>;;NNVC - A high performance Video compression algorithm
;;          - Designed and written by Jared Nelsen
;;               - Began implementation 12-6-18

(ns nnvc.nnvc
  (:gen-class))

;;----------------------------------------------------------------------------------------
;; Current Implementation Notes
;;----------------------------------------------------------------------------------------
;;
;; 1. Currently, the generateLayers function generates N - 1 layers. This may be a stumbling
;;    block when it comes to feeding forward and then reading the result. 
;;
;;----------------------------------------------------------------------------------------
;;----------------------------------------------------------------------------------------
;; Video Data
;;----------------------------------------------------------------------------------------
;;
;; Video data will be kept in a map.
;;
;; Video data has the following properties:
;;
;;    - Video will be kept as a pure vector of integer values partitioned by frame
;;    - Each integer value in the vectors will represent a 32 bit color depth pixel
;;      where each partition of 8 bits represents the r, g, b, and a values of the pixel
;;    - Each frame of video will be a vector line in a file of non-separated rgba values
;;    - Thus the format of a video file will look like
;;
;;      [rgba rgba rgba rgba rgba rgba rgba rgba rgba ... ]
;;      [rgba rgba rgba rgba rgba rgba rgba rgba rgba ... ]
;;      [rgba rgba rgba rgba rgba rgba rgba rgba rgba ... ]
;;      [rgba rgba rgba rgba rgba rgba rgba rgba rgba ... ]
;;      [rgba rgba rgba rgba rgba rgba rgba rgba rgba ... ]
;; 
;;    - Each pixel shade value is stored as an integer in the range 0 to 255 inclusive
;;    - At feed-forward time, the program will read in a frame from disk to rate against
;;       - In practice this looks like using with-open to the file to reference the lazy
;;         stream from disk
;;
;;    - Metadata about the video shall be kept the following format:
;;
;;    Video
;;    -----
;;        [key]           [value]                        
;;    { 
;;      :video-path       String
;;      :frame-count      integer
;;      :frame-width     integer
;;      :frame-height      integer
;;    }
;;                                      
;;----------------------------------------------------------------------------------------

(defn generateFrame
  "Generates a frame of RGBA pixel data. Each RGBA value is a 32 bit pixel where
  the R, G, B, and A values are 8 bits apiece."
  [frame-length]
  (loop [frame '[]
         pixel-count 0]
    (if (< pixel-count frame-length)
      (recur (conj frame (rand-int 256)) (inc pixel-count))
      frame)))

(defn generateNFrames
  "Generates N frames of pixel data."
  [num-frames frame-pixel-count]
  (loop [frames '[]
         frame-count 0]
    (if (< frame-count num-frames)
      (recur (conj frames (generateFrame frame-pixel-count)) (inc frame-count))
      frames)))

(defn packFrames
  "Packs frames into a string to write to a file."
  [frames]
  (loop [accumulator ""
         frames frames]
    (if (not-empty frames)
      (recur (str accumulator (first frames) "\n") (rest frames))
      accumulator)))

(defn generateRandomizedVideoFile
  "Generates and writes a randomized RGBA video file."
  [video-path num-frames frame-height frame-width]
  (let [frames (generateNFrames num-frames (* frame-height frame-width))
        packed-frames (packFrames frames)]
    (with-open [writer (clojure.java.io/writer video-path :append true)]
      (do (.write writer packed-frames)
          {:video-path video-path
           :frame-count num-frames
           :frame-height frame-height
           :frame-width frame-width
           :frame-data frames}))))


;;----------------------------------------------------------------------------------------
;;----------------------------------------------------------------------------------------
;; End Video Data

;;----------------------------------------------------------------------------------------
;; Neural Network
;;----------------------------------------------------------------------------------------
;;
;;    Structure
;;    ---------
;;    - The input neuron will consist of 1 neuron
;;    - There will be N middle layers of a comparatively large number of neurons
;;    - The output layer shall be of the same size as the video data
;;
;;    Weights
;;    -------
;;    - Weights will be in the range of -1 to 1
;;    - Weights will be initialized to very small numbers that are close to zero
;;
;;    Bias
;;    ----
;;    - Bias shall be initialized to very small positive values close to zero
;;    - Bias shall be kept as a single number and added to the layer through addition
;;    - Bias shall be added to each of the neurons before applying the activation function
;;
;;    Activation Functions
;;    --------------------
;;    - The activation function for the hidden layers shall be the ReLU function:
;;          - : max(0, x)
;;    - The activation function for the output layer shall be a specialized function that
;;      scales double values in the range of 0 to 1 to the uniform integer values from 0
;;      to the max 32 bit integer value represented by 4 8 bit Integers each at a value
;;      of 255.
;;
;;    Neurons
;;    -------
;;    - Neurons shall fire in a binary nature: they will be either on or off
;;        - In order to achieve this, the weights must be allowed to be positively and 
;;          negatively valued for use with the ReLU function
;;    - The input neuron will have an integer valued activation, as per the algorithm
;;      specifictions
;;
;;    Schema
;;    ------
;;    - The Neural Network shall be kept in the form of a map:
;;
;;      Matrix form for GPU implementation
;;
;;      Neural Network
;;      --------------
;;            [key]                       [value-type]
;;      {
;;        :input-neuron              integer
;;        :layer-vector              vector of Layers
;;      }
;;
;;            Layer
;;            -----
;;              [key]                    [value-type]
;;            {
;;             :layer-bias           float
;;             :layer-weights        matrix-map
;;             :layer-activations    vector-of-integer-values
;;
;;            Matrix-Map
;;            ----------
;;              [key]                    [value-type]
;;            {
;;             :rows                     integer
;;             :columns                  integer
;;             :matrix-in-vector-form    vector
;;            }
;;    
;;      -------------------------------------
;;      Map form for CPU based implementation:
;;      -------------------------------------
;;
;;      Neural Network
;;      --------------
;;             [key]                    [value-type]
;;      {
;;        :input-neuron-activation   integer
;;        :first-layer-bias          float
;;        :first-hidden-layer        neuron []
;;        :second-layer-bias         float
;;        :second-hidden-layer       neuron []
;;        :output-layer-bias         float
;;        :output-layer              neuron []
;;      }
;;
;;            Neuron
;;            ------
;;                 [key]                [value-type]
;;            {
;;             :activation              float
;;             :incoming-connections    float []
;;            }
;;
;;      Incoming Connections
;;      --------------------
;;        - Weights will be kept in-order, meaning that starting with the first neuron in
;;          the layer, weights will be added to the incoming connections vector
;;          in order with the connections associated with the order of the neurons in the
;;          previous layer. In other words, the incoming connections to a neuron will be
;;          the outgoing weights of each of the neurons in the previous layer that connect
;;          to that neuron.
;;      
;;    
;;----------------------------------------------------------------------------------------

(defn smallRandomValueCloseToZero
  "Generates a small random value that is close to 0 and is either positive or negative."
  []
  (let [positivity (rand-int 2)
        random-value (rand 0.0015)]
    (if (= positivity 0)
      (* random-value -1)
      random-value)))

(defn vectorOfSmallRandomValuesCloseToZero
  "Generates a vector of N values that are close to zero."
  [size]
  (loop [number size 
         vector '[]]
    (if (not= number 0)
     (recur (dec number) (conj vector (smallRandomValueCloseToZero)))
     vector)))

(defn vectorOfZeros
  "Generates a vector of zeros of size N."
  [size]
  (loop [number size 
         vector '[]]
    (if (not= number 0)
     (recur (dec number) (conj vector 0.0))
     vector)))

(defn generateRandomlyInitializedWeightMatrix
  "Generates a randomly initialized matrix with row and column properties."
  [rows columns]
  {:rows rows 
   :columns columns 
   :matrix-in-vector-form (vectorOfSmallRandomValuesCloseToZero (* rows columns))})

(defn generateLayer
  "Generates a single layer in a matrix based Neural Network."
  [rows columns]
  {:layer-bias (smallRandomValueCloseToZero)
   :layer-weights (generateRandomlyInitializedWeightMatrix rows columns)
   :layer-activations (vectorOfZeros (* rows columns))})

(defn generateLayers
  "Generates all layers in the matrix based network.
   The last layer is static at the size of the output vector."
  [input-layer-size hidden-layer-size-vector output-layer-size]
  (let [layer-size-vector-with-first-layer (into [] (cons input-layer-size hidden-layer-size-vector))
        layer-size-vector-with-last-layer (conj layer-size-vector-with-first-layer output-layer-size)]
    (loop [generated-layers '[]
           layer-vector layer-size-vector-with-last-layer
           layer-count (count layer-vector)]
      (if (< (count generated-layers) (dec layer-count))
        (recur (conj generated-layers (generateLayer (first layer-vector) (first (rest layer-vector))))
               (rest layer-vector)
               layer-count)
        (conj generated-layers (generateLayer 1 output-layer-size))))))

(defn generateMatrixBasedNetwork
  "Generates a matrix based Neural Network."
  [input-layer-size hidden-layers-size-vector output-layer-size]
  {:input-neuron-activation 0
   :layer-vector (generateLayers input-layer-size hidden-layers-size-vector output-layer-size)
   :fitness 99999999})

;;----------------------------------------------------------------------------------------
;;----------------------------------------------------------------------------------------
;; End Neural Network


;;----------------------------------------------------------------------------------------
;; Genetic Algorithm
;;----------------------------------------------------------------------------------------
;;
;;    Pseudocode
;;    ----------
;;    1. Initialize population of N members
;;    2. Rate each member of the population
;;    3. Select two members of the population using Roulette Wheel Selection
;;    4. Crossover the two members at a random point according to the crossover rate
;;    5. Mutate the two members according to the mutation rate
;;    6. Repeat steps 3 through 5 until a new population of N members has been produced
;;    7. Repeat procedure until a suitable fitness rating has been acheived
;;
;;    Behaviors
;;    ---------
;;    - Weights will be mutated by randomly picking either positive or negative and then
;;      bumping them in that direction by a very small margin
;;
;;    Roulette Wheel Selection
;;    ------------------------
;;    The algorithm will use roulette wheel selection to perform the selection of two
;;    population members
;;
;;    Conventions
;;    -----------
;;    - Fitness: Fitness is a measure of how good the population member is at solving the
;;               given problem. Because we are minimizing the number of missed integers
;;               in the RGBA vector, we know that the maximal fitness is 0. Thus, we are
;;               minimizing fitness values.
;;    - Ordering: The members of the population will and must be ordered in descending
;;                order in accordance with their fitness values.
;;
;;----------------------------------------------------------------------------------------

;;Constants
;;---------------------------------------------------------------------------

(def CROSSOVER_RATE 0.9)
(def MUTATION_RATE 0.001)

;;End Constants
;;---------------------------------------------------------------------------

;;Initialization
;;---------------------------------------------------------------------------

(defn initializePopulation
  "Initializes the Genetic Algorithm's population."
  ([]
   (initializePopulation 100))
  ([size]
   (initializePopulation size '[]))
  ([size population]
   (if (not= size 0)
     (recur (dec size) (conj population (generateMatrixBasedNetwork 10 2 10)))
     population)))

;;End Initialization
;;---------------------------------------------------------------------------

;;Selection
;;---------------------------------------------------------------------------

(defn extractFitnesses
  "Extracts the fitnesses of a given population.
      - i.e : [333 55 6 5 1]"
  [population]
  (loop [population population
         fitnesses '[]]
    (if (not-empty population)
     (recur (rest population) (conj fitnesses (get (first population) :fitness))) fitnesses)))

(defn computeNormalizedFitnessValues
  "Computes the normalized fitness values for the fitness values passed in."
  ([fitness-values]
   (let [fitness-sum ((fn sumValues
                        [values sum]
                        (loop [values values sum sum]
                          (if (not-empty values)
                            (sumValues (rest values) (+ (first values) sum))
                            sum))) fitness-values 0)]
     (computeNormalizedFitnessValues fitness-values fitness-sum '[])))
  ([fitness-values fitness-sum normalized-values]
   (if (not-empty fitness-values)
     (recur (rest fitness-values)
            fitness-sum 
            (conj normalized-values (float (/ (first fitness-values) fitness-sum))))
     normalized-values)))

(defn computeCumulativeNormalizedFitnessValues
  "Computes the cumulative normalized fitness values for the normalized fitness values
   passed in."
  ([fitness-values]
   (computeCumulativeNormalizedFitnessValues fitness-values
                                             (cons (last fitness-values) '[])
                                             (count fitness-values)
                                             (- (count fitness-values) 2)))
  ([fitness-values new-values original-size index]
   (if (>= index 0)
     (let [current-value (get fitness-values index)
           next-value (first new-values)
           resultant-value (+ current-value next-value)]
       (recur fitness-values 
              (cons resultant-value new-values)
              original-size 
              (dec index)))
     new-values)))

(defn selectParents
  "Performs Roulette Wheel Selection on the population. Returns a vector of
   maps of the form:
    
     {:parentA geonomeA :parentB geonomeB}

   These maps of parents will be used at crossover.
  
   **The population must be sorted in DESCENDING order at pass in time**

   We return a vector of these maps."
([population]
 (let [fitness-values (extractFitnesses population)
       normalized-fitnesses (computeNormalizedFitnessValues fitness-values)
       cumulative-fitness-values (computeCumulativeNormalizedFitnessValues
                                             normalized-fitnesses)]
   (selectParents population cumulative-fitness-values '[]))) 
([population cumulative-fitnesses parent-vector]
 (if (< (count parent-vector) (count population))
   (let [empty-parent-map '{}
         one-parent-map (assoc empty-parent-map 
                               :parentA
                               (selectParents population cumulative-fitnesses parent-vector (rand)))
         two-parent-map (assoc one-parent-map
                               :parentB
                               (selectParents population cumulative-fitnesses parent-vector (rand)))]
   (selectParents population cumulative-fitnesses (conj parent-vector two-parent-map)))
   parent-vector)
 )
 ([population cumulative-fitesses parent-vector random]
  (if (or (> random (last cumulative-fitesses)) (= (count population) 1))
    (last population)
    (recur (drop-last population) (drop-last cumulative-fitesses) parent-vector random))))

;;End Selection
;;---------------------------------------------------------------------------

;;Crossover
;;---------------------------------------------------------------------------

(defn crossoverBias
  "Performs Crossover on the Bias value."
  [bias-A bias-B]
   (if (< (rand) CROSSOVER_RATE)
       bias-B
       bias-A))

(defn crossoverWeights
  "Performs the crossover of the two weight vectors supplied.
   Returns one child weight map.
   weight vectors are guaranteed to be of the same size."
  ([matrix-map-A matrix-map-B]
   (let [weights-A (:matrix-in-vector-form matrix-map-A)
         weights-B (:matrix-in-vector-form matrix-map-B)]
     (assoc matrix-map-A 
            :matrix-in-vector-form (crossoverWeights (count weights-A) weights-A weights-B '[]))))
  ([weight-count weights-A weights-B crossed-over-weights]
   (if (< (count crossed-over-weights) weight-count)
       (if (< (rand) CROSSOVER_RATE)
         (recur weight-count (rest weights-A) (rest weights-B) (conj crossed-over-weights (first weights-B)))
         (recur weight-count (rest weights-A) (rest weights-B) (conj crossed-over-weights (first weights-A))))
       crossed-over-weights)))

(defn crossoverMatrixMap
  "Crosses over the weight vector in the matrix map
   of a layer's weights construct."
  [matrix-map-A matrix-map-B]
  (assoc matrix-map-A
         :matrix-in-vector-form (crossoverWeights matrix-map-A matrix-map-B)))

(defn crossoverLayers
  "Crosses over the components of the two layers passed in.
   A single layer is returned."
  [layerA layerB]
  (assoc layerA 
         :layer-bias (crossoverBias (:layer-bias layerA)
                                    (:layer-bias layerB))
         :layer-weights (crossoverMatrixMap (:layer-weights layerA) 
                                            (:layer-weights layerB))))

(defn crossoverParents
  "Crosses over the weights and biases of the two parents that are passed in.
   A single child geonome will be returned."
  [parentA parentB]
  (loop [parent-A-layers (:layer-vector parentA)
         parent-B-layers (:layer-vector parentB)
         a-layer-from-A (first parent-A-layers)
         a-layer-from-B (first parent-B-layers)
         crossed-over-layers '[]]
      (if (not-empty parent-A-layers)
        (recur (rest parent-A-layers)
               (rest parent-B-layers)
               (first (rest parent-A-layers))
               (first (rest parent-B-layers))
               (conj crossed-over-layers (crossoverLayers a-layer-from-A 
                                                          a-layer-from-B)))
        (assoc parentA :layer-vector crossed-over-layers))))


;;We need to create a new Neural Network type that has been crossed over
;;We must cross over layers in each network in order
;;The algorithm would look like 
;;for each layer. Get the corresponding layer in each parent and cross their
;; biases and matrix maps over.
;;This calls for several functions.`
;; Outer function: for each layer

;;      Neural Network
;;      --------------
;;            [key]                       [value-type]
;;      {
;;        :input-neuron              integer
;;        :layer-vector              vector of Layers
;;      }
;;
;;            Layer
;;            -----
;;              [key]                    [value-type]
;;            {
;;             :layer-bias           float
;;             :layer-weights        matrix-map
;;             :layer-activations    vector-of-integer-values
;;
;;            Matrix-Map
;;            ----------
;;              [key]                    [value-type]
;;            {
;;             :rows                     integer
;;             :columns                  integer
;;             :matrix-in-vector-form    vector
;;            }

(defn crossover
  "Crosses over the weights and biases of the parents that are paired together in
   the input vector.

  The parent vector is comprised of members that look like:

        {:parentA geonomeA :parentB geonomeB}

   Because we are working with two layer networks, the actual numerical
   crossover must take place in two stages for each parent; one for each
   layer of weights.

   We return a vector of geonomes that have been crossed over."
  ([parent-vector]
   (do (prn "Performing Crossover..."))
   (crossover parent-vector '[])) 
  ([parent-vector resultant-vector]
   (if (not-empty parent-vector)
     (let [parents (first parent-vector)
           parentA (:parentA parents)
           parentB (:parentB parents)]
       (recur (rest parent-vector) (conj resultant-vector (crossoverParents parentA parentB))))
     resultant-vector)))

(defn mutateBias
  "Mutates the bias passed in according to the mutation rate."
  [bias]
  (let [random (rand)]
    (if (< random MUTATION_RATE)
      (+ bias (smallRandomValueCloseToZero))
      bias)))

(defn mutateWeights
  "Mutates the weights passed in according to the mutation rate"
  ([weight-map]
   (let [weights (:weight-vector weight-map)]
     (assoc weight-map :weight-vector (mutateWeights weights '[]))))
  ([weight-vector mutated-weight-vector]
   (if (not-empty weight-vector)
     (let [random (rand)
           weight (first weight-vector)
           mutation (smallRandomValueCloseToZero)]
       (if (< random MUTATION_RATE)
         (recur (rest weight-vector) (conj mutated-weight-vector (+ weight mutation)))
         (recur (rest weight-vector) (conj mutated-weight-vector weight))))
     mutated-weight-vector)))

(defn mutatePopulation
  "Mutates the population members."
  ([population]
   (mutatePopulation population '[]))
  ([population resultant-population]
   (if (not-empty population)
     (let [member (first population)

         input-to-hidden-bias (:input-to-hidden-bias member)
         input-to-hidden-weights (:input-to-hidden-weights member)
         hidden-to-hidden-bias (:hidden-to-hidden-bias member)
         hidden-to-hidden-weights (:hidden-to-hidden-weights member)
         hidden-to-output-bias (:hidden-to-output-bias member)
         hidden-to-output-weights (:hidden-to-output-weights member)

         mutated-input-to-hidden-bias (mutateBias input-to-hidden-bias)
         mutated-input-to-hidden-weights (mutateWeights input-to-hidden-weights)
         mutated-hidden-to-hidden-bias (mutateBias hidden-to-hidden-bias)
         mutated-hidden-to-hidden-weights (mutateWeights hidden-to-hidden-weights)
         mutated-hidden-to-output-bias (mutateBias hidden-to-output-bias)
         mutated-hidden-to-output-weights (mutateWeights hidden-to-output-weights)]

     (recur (rest population)
            ;;Overwrite the entries in the member
            (conj resultant-population 
                  (assoc member :input-to-hidden-bias mutated-input-to-hidden-bias
                                :input-to-hidden-weights mutated-input-to-hidden-weights
                                :hidden-to-hidden-bias mutated-hidden-to-hidden-bias
                                :hidden-to-hidden-weights mutated-hidden-to-hidden-weights
                                :hidden-to-output-bias mutated-hidden-to-output-bias
                                :hidden-to-output-weights mutated-hidden-to-output-weights))))
     resultant-population)))

;;End Crossover
;;---------------------------------------------------------------------------

;;Genetic Algorithm Core Functions
;;---------------------------------------------------------------------------


(defn mergeSort
  "Performs Merge Sort on the geonomes passed in descending order by fitness value."
  [geonomes]
  (if (or (empty? geonomes) (= 1 (count geonomes)))
    geonomes
    (let [[leftVector rightVector] (split-at (/ (count geonomes) 2) geonomes)]
      ;recursive call
      (loop [geonome [] leftVector (mergeSort leftVector) rightVector (mergeSort rightVector)]
        ;merging
        (cond (empty? leftVector) (into geonome rightVector)
              (empty? rightVector) (into geonome leftVector)
              :else (if (< 0 (compare 
                              (get (first leftVector) :fitness)
                              (get (first rightVector) :fitness)))
                      (recur (conj geonome (first leftVector)) (rest leftVector) rightVector)
                      (recur (conj geonome (first rightVector)) leftVector (rest rightVector))))))))

(defn ratePopulation
  "Updates the fitnesses of the members of the population based on the fitness function."
  [])

(defn checkForNewBestFitness
  "Checks to see if there is a new best fitness."
  [current-best population]
  (let [new-best (:fitness (last population))]
    (if (< new-best current-best)
      (do (print (str "New best fitness: " new-best))
          (if (= new-best 0)
            (System/exit 0)) 
          new-best)      
      new-best)))

(defn geneticAlgorithm
  "Carries out the genetic algorithm."
  []
  (let [population (initializePopulation)
        best-fitness (:fitness (last population))]
    (loop [population population best-fitness best-fitness]
      (let [rated-population (ratePopulation population)
            sorted-population (mergeSort population)
            fitness (checkForNewBestFitness best-fitness)
            selected-parents (selectParents population)
            crossed-over-population (crossover selected-parents)
            mutated-population (mutatePopulation crossed-over-population)]
        (recur mutated-population fitness)))))


;;----------------------------------------------------------------------------------------
;;----------------------------------------------------------------------------------------
;; End Genetic Algorithm

;;--------------------------------------------------------------------------
;;Utilities
;;---------------------------------------------------------------------------

;;End Utilities
;;---------------------------------------------------------------------------
;;---------------------------------------------------------------------------

;;----------------------------------------------------------------------------------------
;; Applicaton Functions
;;----------------------------------------------------------------------------------------


;;(compress)

;;The entry point for NNVC
;; (defn -main
;;   "The entry point for the NNVC application."
;;   []
;;   (compress))
