 <base>
  <Simulation
    time_granularity={"D"}
    simulation_length={500}
  />
 </base>
 
 <simulation>
   <beta_lag_sickness>0.4</beta_lag_sickness>
   <beta_rain></beta_rain>
 </simulation>

 <simulation base={sim}>
   <beta_lag_sickness>0.4</beta_lag_sickness>
   <beta_rain></beta_rain>
 </simulation>

 <simulation base={sim}>
   <beta_lag_sickness>0.4</beta_lag_sickness>
   <beta_rain></beta_rain>
 </simulation>

 <simulation base={sim}>
   <beta_lag_sickness>0.4</beta_lag_sickness>
   <beta_rain>0.9</beta_rain>
 </simulation>