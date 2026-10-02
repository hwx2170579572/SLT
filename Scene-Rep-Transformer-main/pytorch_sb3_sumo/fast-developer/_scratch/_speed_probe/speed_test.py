import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ["SUMO_HOME"] = r"D:/Program Files (x86)/Eclipse/Sumo"
sys.path.insert(0, os.path.join(os.environ["SUMO_HOME"], "tools"))
import traci

NET = os.path.join(HERE,"t.net.xml"); RTE=os.path.join(HERE,"t.rou.xml"); CFG=os.path.join(HERE,"t.sumocfg")
rte = """<?xml version="1.0" encoding="UTF-8"?>
<routes>
  <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5" maxSpeed="20"/>
  <route id="r0" edges="e0 e1"/>
  <vehicle id="ego" type="car" route="r0" depart="0"/>
</routes>"""
open(RTE,"w").write(rte)
cfg = """<?xml version="1.0" encoding="UTF-8"?>
<configuration>
  <input><net-file value="t.net.xml"/><route-files value="t.rou.xml"/></input>
  <time><begin value="0"/><end value="100"/><step-length value="0.1"/></time>
</configuration>"""
open(CFG,"w").write(cfg)

traci.start(["sumo","-c",CFG,"--no-step-log","true","--duration-log.disable","true"])
conn = traci.getConnection()
for _ in range(5):
    conn.simulationStep()
    if "ego" in conn.vehicle.getIDList(): break
conn.vehicle.setSpeedMode("ego", 0)
conn.vehicle.setSpeed("ego", 10.0); conn.simulationStep()
print("set10 mode0 ->", round(conn.vehicle.getSpeed("ego"),3))
conn.vehicle.setSpeed("ego", 5.0); conn.simulationStep()
print("set5  mode0 ->", round(conn.vehicle.getSpeed("ego"),3))
conn.vehicle.setSpeed("ego", 5.0); conn.simulationStep()
print("hold5 mode0 ->", round(conn.vehicle.getSpeed("ego"),3))
conn.vehicle.setSpeedMode("ego", 31)
conn.vehicle.setSpeed("ego", 10.0); conn.simulationStep()
print("set10 mode31 ->", round(conn.vehicle.getSpeed("ego"),3))
conn.vehicle.setSpeed("ego", 5.0); conn.simulationStep()
print("set5  mode31 ->", round(conn.vehicle.getSpeed("ego"),3))
conn.vehicle.setSpeed("ego", 5.0); conn.simulationStep()
print("set5  mode31 ->", round(conn.vehicle.getSpeed("ego"),3))
traci.close()
