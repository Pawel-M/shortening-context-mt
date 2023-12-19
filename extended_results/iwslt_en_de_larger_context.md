
# IWSLT En-De Extended context results

## BLEU

| Model                         | Context: 4 | Context: 5 | Context: 6 | Context: 7 | Context: 8 | Context: 9 | Context: 10 |
|-------------------------------|------------|------------|------------|------------|------------|------------|-------------|
| Single-encoder                | 10.60      | 24.89      | 1.99       | 1.64       | 1.43       | 1.18       | 0.95        |
| Multi-encoder                 | 28.49      | 28.34      | 27.58      | 26.69      | 25.23      | 8.76       | 7.10        |
| Caching Tokens                | 28.75      | 28.61      | 27.67      | 27.90      | 27.22      | 27.15      | -           |
| Caching Sentence              | 27.87      | 28.30      | 27.55      | 27.67      | 27.20      | -          | -           |
| Shortening - Max Pooling      | 28.32      | 28.42      | 28.15      | 28.06      | -          | -          | -           |
| Shortening - Avg Pooling      | 28.33      | 27.66      | -          | 28.21      | 28.29      | 28.35      | 28.52       |
| Shortening - Linear Pooling   | -          | -          | -          | 28.44      | 28.24      | 28.28      | -           |
| Shortening - Grouping cgrad 1 | 28.79      | 27.80      | 28.65      | 28.18      | 28.09      | 28.17      | 28.39       |
| Shortening - Grouping cgrad 2 | 28.73      | 28.15      | 28.27      | 28.21      | 27.85      | -          | -           |
| Shortening - Selecting        | 28.85      | 28.15      | 27.93      | 28.18      | -          | -          | -           |

## COMET

| Model                         | Context: 4 | Context: 5 | Context: 6 | Context: 7 | Context: 8 | Context: 9 | Context: 10 |
|-------------------------------|------------|------------|------------|------------|------------|------------|-------------|
| Single-encoder                | 0.6266     | 0.7376     | 0.4425     | 0.4253     | 0.3950     | 0.3738     | 0.3597      |
| Multi-encoder                 | 0.7830     | 0.7809     | 0.7692     | 0.7621     | 0.7280     | 0.5682     | 0.5187      |
| Caching Tokens                | 0.7824     | 0.7826     | 0.7773     | 0.7744     | 0.7682     | 0.7560     | -           |
| Caching Sentence              | 0.7766     | 0.7741     | 0.7680     | 0.7680     | 0.7637     | -          | -           |
| Shortening - Max Pooling      | 0.7784     | 0.7782     | 0.7799     | 0.7804     | -          | -          | -           |
| Shortening - Avg Pooling      | 0.7815     | 0.7806     | -          | 0.7812     | 0.7776     | 0.7781     | 0.7814      |
| Shortening - Linear Pooling   | -          | -          | -          | 0.7816     | 0.7780     | 0.7808     | -           |
| Shortening - Grouping cgrad 1 | 0.7799     | 0.7788     | 0.7745     | 0.7796     | 0.7755     | 0.7827     | 0.7811      |
| Shortening - Grouping cgrad 2 | 0.7815     | 0.7808     | 0.7794     | 0.7742     | 0.7785     | -          | -           |
| Shortening - Selecting        | 0.7811     | 0.7793     | 0.7782     | 0.7771     | -          | -          | -           |

## ContraPro Accuracy

| Model                         | Context: 4 | Context: 5 | Context: 6 | Context: 7 | Context: 8 | Context: 9 | Context: 10 |
|-------------------------------|------------|------------|------------|------------|------------|------------|-------------|
| Single-encoder                | 46.09%     | 44.03%     | 43.05%     | 42.07%     | 42.00%     | 38.49%     | 37.03%      |
| Multi-encoder                 | 47.02%     | 44.92%     | 46.25%     | 46.48%     | 43.63%     | 41.53%     | 41.44%      |
| Caching Tokens                | 53.54%     | 47.68%     | 46.88%     | 47.04%     | 45.79%     | 48.15%     | -           |
| Caching Sentence              | 46.57%     | 46.20%     | 44.59%     | 44.91%     | 43.29%     | -          | -           |
| Shortening - Max Pooling      | 51.75%     | 47.13%     | 46.78%     | 46.73%     | -          | -          | -           |
| Shortening - Avg Pooling      | 49.53%     | 49.43%     | -          | 45.88%     | 45.59%     | 46.27%     | 44.66%      |
| Shortening - Linear Pooling   | -          | -          | -          | 46.35%     | 46.90%     | 45.23%     | -           |
| Shortening - Grouping cgrad 1 | 50.24%     | 48.18%     | 47.18%     | 46.27%     | 46.41%     | 45.93%     | 47.86%      |
| Shortening - Grouping cgrad 2 | 49.55%     | 46.06%     | 45.10%     | 47.66%     | 47.19%     | -          | -           |
| Shortening - Selecting        | 47.88%     | 48.98%     | 47.58%     | 45.58%     | -          | -          | -           |
