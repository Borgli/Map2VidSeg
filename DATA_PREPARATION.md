# Datset Preparation

If you want to recreate our dataset or train your own classifier model, the dataset we created is a combination and selected classes of the following datasets:
- [SUN-SEG Dataset](http://sundatabase.org/)
- [HyperKvasir](https://datasets.simula.no/hyper-kvasir/)
- [Kvasir-SEG and Kvasir-Sessile](https://datasets.simula.no/kvasir-seg/)
- [GastroVision](https://datasets.simula.no/gastrovision/)
- [PolypGen](https://github.com/DebeshJha/PolypGen)

We combined the dataset to a binary dataset and balanced the negative examples to have 50% positive and 50% negative examples using the [dataset_analysis.py](dataset_analysis.py) script. 
Using dataset_analysis.py we calculate that we should use a factor of 0.3845 of random negative samples for the sunseg negative dataset in the actual training.

We unfortunately do not provide a script to download or combine the datasets.

The resulting structure when combining should be the following:

```
├──combined-dataset
    ├──gastrovision - 6390 files
        ├──negative - 5272 files
            ├──Angiectasia - 17 files
                ├──1b8c3047-4e18-430f-a171-a988e8e223b0.jpg
                |...
            ├──Barrett's esophagus - 95 files
            ├──Blood in lumen - 171 files
            ├──Cecum - 113 files
            ├──Colon diverticula - 29 files
            ├──Dyed-resection-margins - 246 files
            ├──Erythema - 15 files
            ├──Esophageal varices - 7 files
            ├──Esophagitis - 107 files
            ├──Gastroesophageal_junction_normal z-line - 330 files
            ├──Ileocecal valve - 200 files
            ├──Mucosal inflammation large bowel - 29 files
            ├──Normal esophagus - 140 files
            ├──Normal mucosa and vascular pattern in the large bowel - 1467 files
            ├──Normal stomach - 969 files
            ├──Pylorus - 393 files
            ├──Resection margins - 25 files
            ├──Retroflex rectum - 67 files
            ├──Small bowel_terminal ileum - 846 files
            ├──Ulcer - 6 files
        ├──positive - 1118 files
            ├──Colon polyps - 820 files
                ├──00a5d194-4aa1-495a-afff-5ef6ad6ad3a0.jpg
                |...
            ├──Dyed-lifted-polyps - 141 files
            ├──Gastric polyps - 65 files
            ├──Resected polyps - 92 files
    ├──hyperkvasir - 9653 files
        ├──negative - 7623 files
            ├──barretts - 41 files
                ├──0c9e0051-684f-4cad-9bd2-ed081d876249.jpg
                |...
            ├──barretts-short-segment - 53 files
            ├──bbps-0-1 - 646 files
            ├──bbps-2-3 - 1148 files
            ├──dyed-resection-margins - 989 files
            ├──esophagitis-a - 403 files
            ├──esophagitis-b-d - 260 files
            ├──hemorrhoids - 6 files
            ├──ileum - 9 files
            ├──impacted-stool - 131 files
            ├──pylorus - 999 files
            ├──retroflex-rectum - 391 files
            ├──retroflex-stomach - 764 files
            ├──ulcerative-colitis-grade-0-1 - 35 files
            ├──ulcerative-colitis-grade-1 - 201 files
            ├──ulcerative-colitis-grade-1-2 - 11 files
            ├──ulcerative-colitis-grade-2 - 443 files
            ├──ulcerative-colitis-grade-2-3 - 28 files
            ├──ulcerative-colitis-grade-3 - 133 files
            ├──z-line - 932 files
        ├──positive - 2030 files
            ├──dyed-lifted-polyps - 1002 files
                ├──00a2c35e-97d1-4056-89a5-bc904ff96371.jpg
                |...
            ├──polyps - 1028 files
    ├──kvasir-seg - 1196 files
        ├──negative - empty
        ├──positive - 1196 files
            ├──images - 1000 files
                ├──cju0qkwl35piu0993l0dewei2.jpg
                |...
            ├──sessile - 196 files
    ├──polypgen - 1537 files
        ├──negative - empty
        ├──positive - 1537 files
            ├──images_C1 - 256 files
                ├──100H0050.jpg
                |...
            ├──images_C2 - 301 files
            ├──images_C3 - 457 files
            ├──images_C4 - 227 files
            ├──images_C5 - 208 files
            ├──images_C6 - 88 files
    ├──sunseg - 158690 files
        ├──negative - 109554 files
            ├──case1 - 9961 files
            |...
            ├──case13 - 6328 files
        ├──positive - 49136 files
            ├──case1 - 527 files
            |...
            ├──case100 - 188 files
```