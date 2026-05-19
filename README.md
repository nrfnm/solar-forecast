# ☀️solar-forecast

For a forecasting challenge at [energy-arena.org](https://energy-arena.org)

### Clustering
Sourcing data about installed Solar Capacity via the _open_mastr_ Package from the Markstammdatenregister. Using 
PLZ Coordinate Map Sourced from [WZBSocialScienceCenter](https://github.com/WZBSocialScienceCenter/plz_geocoord) to fill in mostly missing GPS data.
Provided a script to generate K-Means Cluster.

### Integration
To integrate with starer repo from energy-arena.org clone both repos in the same directory side by side and
update .env and copy custom_model.py from this repo to the starer repo.