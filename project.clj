(defproject nnvc "0.1.0"
  :description "A video compression library using Neural Networks and Genetic Algortihms"
  :url "http://example.com/FIXME"
  :license {:name "Copyrighted. Not available for distribution or use."}
  :dependencies [[org.clojure/clojure "1.8.0"]]
  :main ^:skip-aot nnvc.core
  :target-path "target/%s"
  :profiles {:uberjar {:aot :all}}
  :jvm-opts ["-Xmx16g" "-XX:-UseGCOverheadLimit"])
