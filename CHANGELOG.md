# Changelog

## [2.0.0](https://github.com/EBI-Metagenomics/mgnify-bgcs-common-core/compare/mgnify-bgcs-common-core-v1.9.1...mgnify-bgcs-common-core-v2.0.0) (2026-11-08)


### ⚠ BREAKING CHANGES

* **common-core:** LoaderOptions(env_file) defaults to None instead of ".env".
* **common-core:** common_core.metrics and the `ml` extra are gone. Anything installing mgnify-bgcs-common-core[ml] must drop the extra.
* **common-core:** removed modules common_core.chemont, common_core.chemont_classifier, common_core.esmc_embedder, common_core.bgc_region_extractor, common_core.clustering.layout and common_core.clustering.gpu; removed extras esmc, chemont, hpc-gpu and scripts esmc-embed, esmc-embed-bgc, bgc-region-extractor, chemont-classify. read_outputs_tarball no longer returns "coords", partial assignments no longer carry umap_x / umap_y, ClusteringOutputs has no coords / cuda_version / gpu_model, and project_partials no longer takes pri_coords. SCHEMA_VERSION is 3, so older common-core releases reject new input tarballs.

### Features

* **common-core:** add IDF × enrichment weighted-cosine similarity and v2 clustering tarball ([8657305](https://github.com/EBI-Metagenomics/mgnify-bgcs-common-core/commit/86573053a0a88f7ea113221aeece34f4ba89a786))
* **common-core:** compute domain enrichment weights from per-label counts ([3b9179b](https://github.com/EBI-Metagenomics/mgnify-bgcs-common-core/commit/3b9179bab975d7343366bbf58fff6a6bcc94df2d))
* **common-core:** drop CHAMOIS, ESM-C, ChemOnt and the UMAP layout ([6513e72](https://github.com/EBI-Metagenomics/mgnify-bgcs-common-core/commit/6513e7247925cd63442fff7202c2c325297ffb1d))
* **common-core:** load_settings no longer reads ./.env by default ([f8eceab](https://github.com/EBI-Metagenomics/mgnify-bgcs-common-core/commit/f8eceab1823a8740f9d2a082e13f4443027c7a06))
* **common-core:** temper IDF with idf_exponent and use random UMAP init for cosine ([52f98d5](https://github.com/EBI-Metagenomics/mgnify-bgcs-common-core/commit/52f98d576ccf0d4aac4f5d579a3b4473a11467b7))
* **common-core:** vendor the NPClassifier ontology and parse SanntiS 1.0 labels ([c5f12e6](https://github.com/EBI-Metagenomics/mgnify-bgcs-common-core/commit/c5f12e65ab408553a64c3fc523c69eaeaad9b419))


### Bug Fixes

* **common-core:** extract clustering tarballs with filter="data" ([6feeb47](https://github.com/EBI-Metagenomics/mgnify-bgcs-common-core/commit/6feeb4772581416f5d682754e516e465891e3151))
* **common-core:** warn instead of silently ignoring an unreadable .env ([6afc214](https://github.com/EBI-Metagenomics/mgnify-bgcs-common-core/commit/6afc2144f7050f5df6da6999f99b84454b0adb4e))
* **common-core:** zero the similarity diagonal without csr.setdiag ([1627dc6](https://github.com/EBI-Metagenomics/mgnify-bgcs-common-core/commit/1627dc65665e93b565a8da1bfad94850c5136442))
* **contig-filter:** map transcript-keyed eukaryote FAA ids via CDS Parent ([c4fefd4](https://github.com/EBI-Metagenomics/mgnify-bgcs-common-core/commit/c4fefd4b5c039e1cd574df5540414c1d3d2f4158))
* **npc:** add the six "Other &lt;pathway&gt;" superclasses SanntiS 1.0 emits ([cb1d719](https://github.com/EBI-Metagenomics/mgnify-bgcs-common-core/commit/cb1d719aa02a46fd97e4a2819d0cdc9c17601471))


### Performance Improvements

* **common-core:** project partials from the partials × primaries block only ([427d15a](https://github.com/EBI-Metagenomics/mgnify-bgcs-common-core/commit/427d15a0891a4b913cfe81571164c07dbc0e6599))


### Code Refactoring

* **common-core:** delete the metrics package and the ml extra ([47b55e7](https://github.com/EBI-Metagenomics/mgnify-bgcs-common-core/commit/47b55e7d9ebf99550c6fe9ee54e9e347bf2ff2b8))

## [1.9.1](https://github.com/EBI-Metagenomics/mgnify-bgcs-common-core/compare/mgnify-bgcs-common-core-v1.9.0...mgnify-bgcs-common-core-v1.9.1) (2026-10-16)


### Bug Fixes

* **contig-filter:** map FAA proteins to contigs via GFF/GBK CDS features ([f1a5b73](https://github.com/EBI-Metagenomics/mgnify-bgcs-common-core/commit/f1a5b73def941b65fd4f7f6bc2bab9eb0870b333))
* **contig-filter:** map FAA proteins to contigs via GFF/GBK CDS features ([a4df8aa](https://github.com/EBI-Metagenomics/mgnify-bgcs-common-core/commit/a4df8aa4b0a2216b7688b8077eb29194b699117f))

## [1.9.0](https://github.com/EBI-Metagenomics/mgnify-bgcs-common-core/compare/mgnify-bgcs-common-core-v1.8.0...mgnify-bgcs-common-core-v1.9.0) (2026-07-07)


### Features

* **metrics:** add shared BGC-detection metric core ([8c45476](https://github.com/EBI-Metagenomics/mgnify-bgcs-common-core/commit/8c45476d6515a07f25178b60e0c803d5e5f2ede5))


### Bug Fixes

* **clustering:** chunk SpGEMM in dice_similarity_gpu ([16f612a](https://github.com/EBI-Metagenomics/mgnify-bgcs-common-core/commit/16f612a77dfb1f37c3a8d7e2aa4ea59a40f9236b))
* **clustering:** fuse per-block composite to bound GPU memory ([e4a69a7](https://github.com/EBI-Metagenomics/mgnify-bgcs-common-core/commit/e4a69a750982c6e797714a32c6c1a8d31550f6bc))
* **clustering:** symmetrise GPU composite-Dice on host ([0e9b560](https://github.com/EBI-Metagenomics/mgnify-bgcs-common-core/commit/0e9b56079499abb15f08520b6dfc72d3df7ff374))
* **gbk:** unwrap wrapped identifier qualifiers after SeqIO.write ([2d3e07c](https://github.com/EBI-Metagenomics/mgnify-bgcs-common-core/commit/2d3e07c451934a5211481fadae07660c21accb51))

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
