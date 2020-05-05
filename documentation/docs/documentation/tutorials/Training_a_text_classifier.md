# Training a short text classifier of German business names

In this tutorial we will train a basic short-text classifier for predicting the sector of a business based only on its business name. For this we will use a training dataset with business names and business categories in German.

The tutorial will guide you through the following steps:


[[toc]]



## Explore and prepare training and evaluation data

Let's take a look at the data we will use for training.


```python
import pandas as pd
```


```python
df = pd.read_csv('data/business.cat.10k.csv')
```


```python
df.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>label</th>
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>44310</td>
      <td>Tiefbau</td>
      <td>Baugeschäft Haßmann Gmbh Wörblitz</td>
    </tr>
    <tr>
      <th>1</th>
      <td>39433</td>
      <td>Restaurants</td>
      <td>Gaststätten, Restaurants - Sucos Do Brasil Coc...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8828</td>
      <td>Autowerkstätten</td>
      <td>Lankes Kfz-werkstatt</td>
    </tr>
    <tr>
      <th>3</th>
      <td>61668</td>
      <td>Werbeagenturen</td>
      <td>Feine Reklame Gesellschaft Für Strategische Kr...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>34837</td>
      <td>Maler</td>
      <td>Müller Vladimir &amp; Co. Malermeister</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1478</td>
      <td>Allgemeinärzte</td>
      <td>Renninger Arztpraxis Für Allgemeinmedizin Dr.</td>
    </tr>
    <tr>
      <th>6</th>
      <td>21584</td>
      <td>Friseure</td>
      <td>Coiffeur La Vie</td>
    </tr>
    <tr>
      <th>7</th>
      <td>34259</td>
      <td>Maler</td>
      <td>Kiesewalter Malermeister Thomas</td>
    </tr>
    <tr>
      <th>8</th>
      <td>10951</td>
      <td>Dienstleistungen</td>
      <td>Gerhard Pflaum Minden-herforder-verkehrs-servi...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>36476</td>
      <td>Physiotherapie</td>
      <td>Hellriegel - Thoms - Feliksßen Rückenzentrum K...</td>
    </tr>
  </tbody>
</table>
</div>



As we can see we have two relevant columns `label` and `text`. 

Our classifier will be trained to predict the `label` given a `text`.

Let's check the distribution of our `label` columns


```python
pd.DataFrame(df.label.value_counts())
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Unternehmensberatungen</th>
      <td>775</td>
    </tr>
    <tr>
      <th>Friseure</th>
      <td>705</td>
    </tr>
    <tr>
      <th>Tiefbau</th>
      <td>627</td>
    </tr>
    <tr>
      <th>Dienstleistungen</th>
      <td>613</td>
    </tr>
    <tr>
      <th>Gebrauchtwagen</th>
      <td>567</td>
    </tr>
    <tr>
      <th>Restaurants</th>
      <td>526</td>
    </tr>
    <tr>
      <th>Architekturbüros</th>
      <td>523</td>
    </tr>
    <tr>
      <th>Elektriker</th>
      <td>513</td>
    </tr>
    <tr>
      <th>Vereine</th>
      <td>488</td>
    </tr>
    <tr>
      <th>Versicherungsvermittler</th>
      <td>462</td>
    </tr>
    <tr>
      <th>Sanitärinstallationen</th>
      <td>416</td>
    </tr>
    <tr>
      <th>Edv</th>
      <td>413</td>
    </tr>
    <tr>
      <th>Maler</th>
      <td>413</td>
    </tr>
    <tr>
      <th>Physiotherapie</th>
      <td>366</td>
    </tr>
    <tr>
      <th>Werbeagenturen</th>
      <td>362</td>
    </tr>
    <tr>
      <th>Apotheken</th>
      <td>346</td>
    </tr>
    <tr>
      <th>Vermittlungen</th>
      <td>345</td>
    </tr>
    <tr>
      <th>Hotels</th>
      <td>343</td>
    </tr>
    <tr>
      <th>Autowerkstätten</th>
      <td>332</td>
    </tr>
    <tr>
      <th>Elektrotechnik</th>
      <td>329</td>
    </tr>
    <tr>
      <th>Allgemeinärzte</th>
      <td>274</td>
    </tr>
    <tr>
      <th>Handelsvermittler Und -vertreter</th>
      <td>262</td>
    </tr>
  </tbody>
</table>
</div>



## Configure your `biome.text` Pipeline


```python
from biome.text.api_new import Pipeline
from biome.text.api_new.configuration import TrainerConfiguration
from biome.text.api_new.helpers import yaml_to_dict
```

### Pipeline configuration from YAML

A `biome.text` pipeline has the following main components:

```yaml
name: # the name

tokenizer: # how to tokenize input text

features: # this input features of the model

encoder: # the backbone model encoder

head: # your task configuration


```

Our complete configuration for this tutorial is:

```yaml
name: german-business-categories

features:
    words:
        embedding_dim: 100
        lowercase_tokens: true
    chars:
        embedding_dim: 8
        encoder:
            type: cnn
            num_filters: 50
            ngram_filter_sizes: [ 4 ]
        dropout: 0.2

encoder:
    hidden_size: 512
    num_layers: 2
    dropout: 0.5
    type: lstm

head:
    type: TextClassification
    pooler:
        type: boe
    labels: ['Allgemeinärzte', 'Apotheken', 'Architekturbüros',
             'Autowerkstätten', 'Dienstleistungen', 'Edv', 'Elektriker',
             'Elektrotechnik', 'Friseure', 'Gebrauchtwagen',
             'Handelsvermittler Und -vertreter', 'Hotels', 'Maler',
             'Physiotherapie', 'Restaurants', 'Sanitärinstallationen',
             'Tiefbau', 'Unternehmensberatungen', 'Vereine', 'Vermittlungen',
             'Versicherungsvermittler', 'Werbeagenturen']
```


```python
pl = Pipeline.from_file("configs/text_classifier.yml")
```


```python
pl.config.as_dict()
```




    {'name': 'business-categories',
     'tokenizer': {'lang': 'en',
      'skip_empty_tokens': False,
      'max_sequence_length': None,
      'max_nr_of_sentences': None,
      'text_cleaning': None,
      'segment_sentences': False},
     'features': {'words': {'embedding_dim': 100, 'lowercase_tokens': True},
      'chars': {'embedding_dim': 8,
       'encoder': {'type': 'cnn', 'num_filters': 50, 'ngram_filter_sizes': [4]},
       'dropout': 0.2}},
     'encoder': {'hidden_size': 512,
      'num_layers': 2,
      'dropout': 0.5,
      'type': 'lstm',
      'input_size': 150},
     'head': {'type': 'TextClassification',
      'pooler': {'type': 'boe'},
      'labels': ['Allgemeinärzte',
       'Apotheken',
       'Architekturbüros',
       'Autowerkstätten',
       'Dienstleistungen',
       'Edv',
       'Elektriker',
       'Elektrotechnik',
       'Friseure',
       'Gebrauchtwagen',
       'Handelsvermittler Und -vertreter',
       'Hotels',
       'Maler',
       'Physiotherapie',
       'Restaurants',
       'Sanitärinstallationen',
       'Tiefbau',
       'Unternehmensberatungen',
       'Vereine',
       'Vermittlungen',
       'Versicherungsvermittler',
       'Werbeagenturen']}}



### Testing our pipeline before training

It recommended to check that our pipeline is correctly setup using the `predict` method.

::: warning

Our pipeline has not been trained before, so its weights are random. Do not expect its predictions to make sense for now.

:::



```python
pl.predict('Some text')
```




    {'logits': array([-0.0333772 , -0.01114595,  0.08185824,  0.00720856, -0.01808064,
             0.0209163 , -0.04119281,  0.0234425 ,  0.00120479,  0.04529068,
            -0.02560528,  0.03243363, -0.02825472,  0.01238234,  0.00707909,
            -0.05999601,  0.05878261,  0.03128546, -0.01267068,  0.00673078,
             0.01568662,  0.02453783], dtype=float32),
     'probs': array([0.04366268, 0.04464422, 0.04899553, 0.04547121, 0.0443357 ,
            0.04609881, 0.04332276, 0.04621541, 0.04519903, 0.04723625,
            0.04400334, 0.04663282, 0.04388691, 0.04570708, 0.04546533,
            0.04251576, 0.04787787, 0.04657931, 0.04457621, 0.04544949,
            0.04585836, 0.04626606], dtype=float32),
     'classes': {'Architekturbüros': 0.04899553209543228,
      'Tiefbau': 0.047877874225378036,
      'Gebrauchtwagen': 0.04723624885082245,
      'Hotels': 0.04663281515240669,
      'Unternehmensberatungen': 0.046579305082559586,
      'Werbeagenturen': 0.046266064047813416,
      'Elektrotechnik': 0.04621541127562523,
      'Edv': 0.04609880968928337,
      'Versicherungsvermittler': 0.04585836082696915,
      'Physiotherapie': 0.04570708051323891,
      'Autowerkstätten': 0.04547121003270149,
      'Restaurants': 0.04546532779932022,
      'Vermittlungen': 0.04544949159026146,
      'Friseure': 0.04519902914762497,
      'Apotheken': 0.04464422166347504,
      'Vereine': 0.04457620531320572,
      'Dienstleistungen': 0.04433570057153702,
      'Handelsvermittler Und -vertreter': 0.04400334134697914,
      'Maler': 0.04388691112399101,
      'Allgemeinärzte': 0.04366267845034599,
      'Elektriker': 0.04332275688648224,
      'Sanitärinstallationen': 0.04251576215028763},
     'max_class': 'Architekturbüros',
     'max_class_prob': 0.04899553209543228,
     'label': 'Architekturbüros',
     'prob': 0.04899553209543228}




```python
yaml_to_dict("configs/trainer.yml")
```




    {'batch_size': 64,
     'num_epochs': 100,
     'optimizer': {'type': 'adam', 'lr': 0.01},
     'validation_metric': '-loss',
     'patience': 2}




```python
trainer = TrainerConfiguration(**yaml_to_dict("configs/trainer.yml"))
```


```python

```
