#!/usr/bin/env python3
import os, json, requests
from pathlib import Path
from neo4j import GraphDatabase

DATA_DIR = Path("data/parsed")

NEO4J_URI  = os.environ["NEO4J_URI"]   # bolt+s://... OR neo4j+s://...
NEO4J_USER = os.environ["NEO4J_USER"]
NEO4J_PASS = os.environ["NEO4J_PASS"]
GEMINI_KEY = os.environ["GEMINI_API_KEY"]

PROMPT = """Extract from the paragraph:
Return JSON:
{
 "entities":[
   {"type":"Exposure|Intervention|Organism|Tissue|Outcome|Biomarker|MissionConstraint","name":"...", "value":null, "unit":null, "id":null}
 ],
 "relations":[
   {"type":"UNDER|APPLIES|MEASURES|AFFECTS|MODULATES|INDICATES",
    "head":{"type":"...", "name":"..."},
    "tail":{"type":"...", "name":"..."},
    "evidence_sentence":"...", "direction": "increase|decrease|mixed|null",
    "magnitude": "12%|null", "pval": "0.03|null"}
 ]
}
Only include items clearly supported by the text. Keep names concise.
"""

def gemini_ie(text):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent?key={GEMINI_KEY}"
    body = {
      "contents":[{"parts":[{"text": PROMPT+"\n\nPARAGRAPH:\n"+text[:8000]}]}],
      "generationConfig":{"responseMimeType":"application/json"}
    }
    r = requests.post(url, json=body, timeout=30)
    r.raise_for_status()
    out = r.json()
    txt = out["candidates"][0]["content"]["parts"][0]["text"]
    return json.loads(txt)

def upsert_graph(tx, paper_id, section, idx, ents, rels):
    tx.run("MERGE (p:Paper {paper_id:$pid})", pid=paper_id)
    for e in ents:
        tx.run(f"MERGE (n:{e['type']} {{name:$name}}) "
               f"ON CREATE SET n.created=timestamp() "
               f"SET n.id=$id, n.unit=$unit, n.value=$value",
               name=e["name"], id=e.get("id"), unit=e.get("unit"), value=e.get("value"))
    for r in rels:
        tx.run(f"""
        MATCH (h:{r['head']['type']} {{name:$hname}}), (t:{r['tail']['type']} {{name:$tname}})
        MERGE (h)-[rel:{r['type']}]->(t)
        ON CREATE SET rel.created=timestamp()
        SET rel.paper_id=$pid, rel.section=$sec, rel.para_idx=$idx,
            rel.evidence=$ev, rel.direction=$dir, rel.magnitude=$mag, rel.pval=$pval
        """, hname=r["head"]["name"], tname=r["tail"]["name"], pid=paper_id, sec=section,
           idx=idx, ev=r.get("evidence_sentence"), dir=r.get("direction"),
           mag=r.get("magnitude"), pval=r.get("pval"))

def main():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
    with driver.session() as sess:
        for p in sorted(DATA_DIR.glob("*.jsonl")):
            paper_id = p.stem
            with p.open() as f:
                for line in f:
                    rec = json.loads(line)
                    # Start with key sections to save tokens
                    if str(rec["section"]).lower() not in {"abstract","results","discussion","conclusion","conclusions"}:
                        continue
                    try:
                        ie = gemini_ie(rec["text"])
                    except Exception:
                        continue
                    ents = ie.get("entities", [])
                    rels = ie.get("relations", [])
                    if not ents and not rels:
                        continue
                    sess.execute_write(upsert_graph, rec["paper_id"], rec["section"], rec["para_idx"], ents, rels)
            print(f"✓ Graph updated: {paper_id}")
    driver.close()

if __name__ == "__main__":
    main()