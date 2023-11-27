# Qualitative analysis of the translations on the IWSLT 2017 en-de testset

test line 32
```
En:
This is a nice building.
But it doesn't have much to do with what a library actually does today.

De:
Aber es hat nicht viel mit dem zu tun, was eine Bibliothek heute leistet.

Transformer Pred:
Aber es hat nicht viel damit zu tun, was eine Bibliothek heute tut.

Shortening - Grouping (ctx 1) Pred:
Aber es hat nicht viel mit dem zu tun, was eine Bibliothek heute tut.

Shortening - Selecting (ctx 1) Pred:
Aber es hat nicht viel mit der heutigen Bibliothek zu tun.
```


test line 257

```
En:
Zak Ebrahim is not my real name.
I changed it when my family decided to end our connection with my father and start a new life.

De:
Ich habe ihn geändert, als meine Familie beschloss, den Kontakt zu meinem Vater abzubrechen und ein neues Leben zu beginnen.

Transformer Pred - incorrect:
Ich änderte es, als meine Familie entschied, unsere Verbindung mit meinem Vater zu beenden und ein neues Leben zu starten.

Shortening - Grouping (ctx 1) Pred:
Ich habe ihn verändert, als meine Familie entschied, unsere Verbindung mit meinem Vater zu beenden und ein neues Leben zu beginnen.

Shortening - Selecting (ctx 1) Pred - incorrect:
Ich habe es verändert, als meine Familie beschloss, unsere Verbindung mit meinem Vater zu beenden und ein neues Leben zu beginnen.
```

test line 269

```
En:
Your movie has smell and taste and touch.
It has a sense of your body, pain, hunger, orgasms.

De:
Er hat ein Gefühl für Ihren Körper, Schmerz, Hunger, Orgasmen.

Transformer Pred - incorrect:
Es hat ein Gefühl von Körper, Schmerz, Hunger, Organigmen.

Shortening - Grouping (ctx 1) Pred:
Er hat ein Gefühl Ihres Körpers, des Schmerzes, des Hungers, des Organigas.

Shortening - Selecting (ctx 1) Pred:
Er hat ein Gefühl Ihres Körpers, von Schmerz, Hunger, Organigmen.
```

test line 298

```
En:
And this work has been wonderful. It's been great.
But it also has some fundamental limitations so far.

De:
Aber sie hat auch noch immer einige grundlegende Grenzen.

Transformer Pred - incorrect:
Aber es hat bis jetzt auch einige fundamentale Grenzen.

Shortening - Grouping (ctx 1) Pred - incorrect:
Aber es hat bis jetzt noch grundlegende Grenzen.

Shortening - Selecting (ctx 1) Pred:
Aber sie hat auch bis jetzt einige fundamentale Grenzen.
```