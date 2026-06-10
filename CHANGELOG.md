# Changelog

## [1.8.0](https://github.com/EBI-Metagenomics/mgnify-bgcs-common-core/compare/mgnify-bgcs-common-core-v1.7.0...mgnify-bgcs-common-core-v1.8.0) (2026-06-10)


### Features

* **chemical searches:** Support coverage_similarity for ChemOnt term similarity ([c215339](https://github.com/EBI-Metagenomics/mgnify-bgcs-common-core/commit/c21533940ff4eae5100ffa347f166da9a35d106d))

## [1.7.0](https://github.com/EBI-Metagenomics/mgnify-bgcs-common-core/compare/mgnify-bgcs-common-core-v1.6.1...mgnify-bgcs-common-core-v1.7.0) (2026-06-08)


### Features

* Support ClassyFire using API ([80dbbd6](https://github.com/EBI-Metagenomics/mgnify-bgcs-common-core/commit/80dbbd69b722e0e541e73c76e1143a29f7dacf44))

## [1.6.1](https://github.com/EBI-Metagenomics/mgnify-bgcs-common-core/compare/mgnify-bgcs-common-core-v1.6.0...mgnify-bgcs-common-core-v1.6.1) (2026-06-05)


### Bug Fixes

* classify antiSMASH base-category names and missing RiPP types ([61efc10](https://github.com/EBI-Metagenomics/mgnify-bgcs-common-core/commit/61efc10e455c43365d5c82538eee2a3874bb2db0))
* classify antiSMASH base-category names and missing RiPP types ([b2fe722](https://github.com/EBI-Metagenomics/mgnify-bgcs-common-core/commit/b2fe722d3c253995efeb41ee9cb38bfce19b86db))

## [1.6.0](https://github.com/EBI-Metagenomics/mgnify-bgcs-common-core/compare/mgnify-bgcs-common-core-v1.5.0...mgnify-bgcs-common-core-v1.6.0) (2026-06-05)


### Features

* Support BGC class standardization for all tools ([0943f40](https://github.com/EBI-Metagenomics/mgnify-bgcs-common-core/commit/0943f40f18d217f59f56ff43ca9b59bf9335abc5))

## [1.5.0](https://github.com/EBI-Metagenomics/mgnify-bgcs-common-core/compare/mgnify-bgcs-common-core-v1.4.1...mgnify-bgcs-common-core-v1.5.0) (2026-06-02)


### Features

* **clustering:** Support compact naming of GCF/clusters ([9439e14](https://github.com/EBI-Metagenomics/mgnify-bgcs-common-core/commit/9439e14f69416828b4bd727201b8a8d82fcdc29e))

## [1.4.1](https://github.com/EBI-Metagenomics/mgnify-bgcs-common-core/compare/mgnify-bgcs-common-core-v1.4.0...mgnify-bgcs-common-core-v1.4.1) (2026-06-02)


### Bug Fixes

* **clustering:** apply nrb→ibgc rename to match portal Phase 2f contract ([11f1862](https://github.com/EBI-Metagenomics/mgnify-bgcs-common-core/commit/11f186225358758e6034074f873cbd24dd2ec441))

## [1.4.0](https://github.com/EBI-Metagenomics/mgnify-bgcs-common-core/compare/mgnify-bgcs-common-core-v1.3.0...mgnify-bgcs-common-core-v1.4.0) (2026-06-02)


### Features

* **bgc clustering:** Support local clustering for portal deployment ([5fdd0bf](https://github.com/EBI-Metagenomics/mgnify-bgcs-common-core/commit/5fdd0bf183d8c99f037651b44fa0f22e26ecb823))
* **NP preds:** Support CHEMONT to GENE using CHAMOIS results ([5efc45c](https://github.com/EBI-Metagenomics/mgnify-bgcs-common-core/commit/5efc45c474c1e9669feea4e04777ac79269f693a))


### Bug Fixes

* **clustering:** score validated NRB novelty as 0 via decoupled validated-similarity block ([7bb5ceb](https://github.com/EBI-Metagenomics/mgnify-bgcs-common-core/commit/7bb5ceb7d4e9d5c11af24446468524acdc3223c7))
* **gpu clustering:** Add array ref to sim_gpu.setdiag ([bf0cf17](https://github.com/EBI-Metagenomics/mgnify-bgcs-common-core/commit/bf0cf171b1edf5caeb79b5ea8117d5f7550166dd))
* **NP predictions:** gene - chemont asociation scores ([1b00d31](https://github.com/EBI-Metagenomics/mgnify-bgcs-common-core/commit/1b00d3104683972ae4b9117533b9d667427e25b6))


### Performance Improvements

* **bgc_aggregator:** Change esm model from 600M to 300M params ([62294d9](https://github.com/EBI-Metagenomics/mgnify-bgcs-common-core/commit/62294d9f755e109296fb6223f38df19e332d9114))

## [1.3.0](https://github.com/EBI-Metagenomics/mgnify-bgcs-common-core/compare/mgnify-bgcs-common-core-v1.2.0...mgnify-bgcs-common-core-v1.3.0) (2026-04-13)


### Features

* **bgc_vector:** Support a central protein/BGC embedding module ([0bbb726](https://github.com/EBI-Metagenomics/mgnify-bgcs-common-core/commit/0bbb7266b95742ec8ab219b55db0b8795cbbe6dc))
* ChemOnt classifier ([dfcbbe1](https://github.com/EBI-Metagenomics/mgnify-bgcs-common-core/commit/dfcbbe1473ecaf3b32bf78e63e81dc4b633f4466))
* Contig length filter ([ac6abd4](https://github.com/EBI-Metagenomics/mgnify-bgcs-common-core/commit/ac6abd410e42b3bf22d2abc4a0ec37bd0f5a6390))
* **utilities:** Support contig filter, bgc embeddings with positional encoding, and region extraction ([745359b](https://github.com/EBI-Metagenomics/mgnify-bgcs-common-core/commit/745359ba6c7746419bf3480f77f77fe050dc5005))


### Bug Fixes

* **chemont_classifier:** Reformat output to match bgc_portal pattern ([693e545](https://github.com/EBI-Metagenomics/mgnify-bgcs-common-core/commit/693e545af6e2e3ef9426870ef4c380b9c76608bc))
* **ci:** stop tracking build artifacts ([e4e8565](https://github.com/EBI-Metagenomics/mgnify-bgcs-common-core/commit/e4e8565e796d285e321833a3d852e358fc7bd2a9))

## [1.2.0](https://github.com/Finn-Lab/mgnify-bgcs-common-core/compare/mgnify-bgcs-common-core-v1.1.0...mgnify-bgcs-common-core-v1.2.0) (2026-02-23)


### Features

* **versioning:** add helper to get version from packages ([b90b7b6](https://github.com/Finn-Lab/mgnify-bgcs-common-core/commit/b90b7b666644eb3e40809aa26aec676578563eda))
* **workers:** add worker template for future reference ([476f861](https://github.com/Finn-Lab/mgnify-bgcs-common-core/commit/476f86130202e870030058fb90ba7ba6b0e4cb5c))


### Bug Fixes

* **BGC class normalization:** retrun class if not in class map for normalization ([437dc4c](https://github.com/Finn-Lab/mgnify-bgcs-common-core/commit/437dc4c651e9e682c04cf1c06c4062932856d287))
* **logging:** resolve attribute shadowing of pydantic BaseMolde ([100394c](https://github.com/Finn-Lab/mgnify-bgcs-common-core/commit/100394c5f906484e2abe228693bda8b3717c65a2))
* **worker:** add worker template ([ad120d3](https://github.com/Finn-Lab/mgnify-bgcs-common-core/commit/ad120d38f6bd98508933029a2108e54631c33802))

## [1.1.0](https://github.com/Finn-Lab/mgnify-bgcs-common-core/compare/mgnify-bgcs-common-core-v1.0.0...mgnify-bgcs-common-core-v1.1.0) (2026-02-16)


### Features

* **ci:** support release please auto bump ([7a67277](https://github.com/Finn-Lab/mgnify-bgcs-common-core/commit/7a67277df527867ccbb92680914516238d964524))
