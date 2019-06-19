(ns nnvc.core-test
  (:require [clojure.test :refer :all]
            [nnvc.core :refer :all]
			[nnvc.nnvc :refer :all]))

;;----------------------------------------------------------------------------------------
;; Neural Network
;;----------------------------------------------------------------------------------------

(def CLOSE_TO_ZERO_LOWER_BOUND -0.0015)
(def CLOSE_TO_ZERO_UPPER_BOUND 0.0015)

(deftest smallRandomValueCloseToZero_test
  (testing "Value out of range..."
    (let [value (smallRandomValueCloseToZero)]
      (and (is (>= value CLOSE_TO_ZERO_LOWER_BOUND)) (is (<= value CLOSE_TO_ZERO_UPPER_BOUND))))))

(deftest vectorOfSmallRandomValuesCloseToZero_test
  (testing "Length and some value out of range..."
    (loop [size 15
           values (vectorOfSmallRandomValuesCloseToZero size)
           iterations 0]
      (if (= size iterations)
        (is (= (count values) size))
        (let [value (first values)]
          (and (is (>= value CLOSE_TO_ZERO_LOWER_BOUND)) (is (<= value CLOSE_TO_ZERO_UPPER_BOUND)))
          (recur size values (inc iterations)))))))

(deftest vectorOfZeros_test
  (testing "Length and all values are zero..."
    (loop [size 15
           values (vectorOfZeros size)
           iterations 0]
      (if (= size iterations)
        (is (= (count values) size))
        (let [value (first values)]
          (is (= value 0.0))
          (recur size values (inc iterations)))))))

(deftest generateRandomlyInitializedWeightMatrix_test
  (testing "Size of weight vector and value range..."
    (let [length 4
          width 6
          matrix (generateRandomlyInitializedWeightMatrix length width)]
      (loop [elements (:weight-vector matrix)
             iterations 0
             size (count elements)]
        (if (= size iterations)
          (and (is (= size (count elements))) (is (= size (* length width))))
          (let [value (first elements)]
            (and (is (>= value CLOSE_TO_ZERO_LOWER_BOUND)) (is (<= value CLOSE_TO_ZERO_UPPER_BOUND)))
            (recur elements (inc iterations) size)))))))

(deftest generateMatrixBasedNetwork_test
  (testing "Number of elements in the Network, sizes of layers, and fitness level..."
    (let [first-layer-size 3
          second-layer-size 4
          output-layer-size 5         
          network (generateMatrixBasedNetwork first-layer-size second-layer-size output-layer-size)
          
          detected-first-hidden-layer-size (count (:first-hidden-layer network))
          detected-second-hidden-layer-size (count (:second-hidden-layer network))
          detected-output-layer-size (count (:output-layer network))]

      (and (is (= first-layer-size detected-first-hidden-layer-size))
           (is (= second-layer-size detected-second-hidden-layer-size))
           (is (= output-layer-size detected-output-layer-size))
           (is (> (:fitness network) 1000))))))

;;----------------------------------------------------------------------------------------
;;----------------------------------------------------------------------------------------
;; End Neural Network

;;----------------------------------------------------------------------------------------
;; Genetic Algorithm
;;----------------------------------------------------------------------------------------

(deftest initializePopulation_test
  (testing "The size of the population is correct..."
    (let [size 10
          population (initializePopulation size)]
      (is (= (count population) size)))))

(deftest initializeGAContext_test
  (testing "Features of the GA Context..."
    (let [size 20
          default-context (initializeGAContext)
          parameterized-context (initializeGAContext size)]
      (and (is (= (count (:population default-context)) 100))
           (is (= (count (:population parameterized-context)) size))))))

(deftest extractFitnesses_test
  (testing "Successful fitness vector extraction..."
    (let [population (initializePopulation 10)
          fitnesses (extractFitnesses population)]
      (is (vector? fitnesses)))))

(defn sortedOrder_test
  "Tests if each element is greater than the last..."
  [values]
  (if (<= 2 (count values))
    (let [this (first values)
          next (nth values 1)]
      (is (>= this next))
      (recur (rest values)))))

(deftest computeNormalizedFitnessValues_test
  (testing "Each value is less than the last..."
    (let [values '[33 22 16 11 3 0.5]
          normalized-values (computeNormalizedFitnessValues values)]
      (sortedOrder_test normalized-values))))

;;----------------------------------------------------------------------------------------
;;----------------------------------------------------------------------------------------
;; End Genetic Algorithm

;;--------------------------------------------------------------------------
;;Utilities
;;---------------------------------------------------------------------------

(defn sortedPopulation_test
  [population]
     (if (<= 2 (count population))
       (let [this (first population)
             next (nth population 1)]
         (is (>= (:fitness this) (:fitness next)))
         (recur (rest population)))))

(deftest mergeSort_test
  (testing "Merge sort efficacy..."
    (let [population (initializePopulation 100)
          random-fitness-population ((fn randomize-fitnesses
                                       [population new-population]
                                       (if (not-empty population)
                                         (let [member (first population)
                                               changed-member (assoc member :fitness (rand-int 1000))]
                                           (randomize-fitnesses
                                            (rest population)
                                            (conj new-population changed-member)))
                                         new-population)) population '[])
          sorted-population (mergeSort random-fitness-population)]
      (sortedPopulation_test sorted-population))))



;;End Utilities
;;---------------------------------------------------------------------------
;;---------------------------------------------------------------------------
