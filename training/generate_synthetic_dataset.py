"""
Synthetic Dataset Generator for StartupTN Proposal Evaluation LLM Fine-Tuning.

Generates 500+ high-quality, diverse startup proposal examples across Tamil Nadu
sectors with realistic evaluation outputs following the ProposalEvaluationOutput schema.

Usage:
    python training/generate_synthetic_dataset.py --count 600 --output data/processed/startup_proposals_synthetic.jsonl
    python training/generate_synthetic_dataset.py --count 600 --output data/processed/startup_proposals_synthetic.jsonl --seed 42
"""

import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Any

SYSTEM_PROMPT = (
    "You are an expert Startup Proposal Evaluation AI assisting official evaluators "
    "at StartupTN (Government of Tamil Nadu).\n"
    "Your task is to analyze startup proposals submitted for funding, incubation, "
    "mentoring, or ecosystem support.\n\n"
    "Analyze the given proposal text carefully and produce a structured assessment "
    "in valid JSON format."
)

REC_PROCEED   = "PROCEED_TO_HUMAN_EVALUATION"
REC_REJECT    = "REJECT_HIGH_RISK"
REC_MORE_INFO = "REQUEST_MORE_INFORMATION"
REC_MONITOR   = "CONDITIONAL_APPROVAL_WITH_MONITORING"

TN_CITIES = [
    "Chennai","Coimbatore","Madurai","Tiruchirappalli","Salem",
    "Tiruppur","Vellore","Erode","Thoothukudi","Tirunelveli",
    "Dindigul","Thanjavur","Sivakasi","Hosur","Karur",
    "Namakkal","Cuddalore","Kanchipuram","Kumbakonam","Nagercoil",
]

def _pick(pool): return random.choice(pool)
def _ri(a,b): return random.randint(a,b)
def _rf(a,b,d=2): return round(random.uniform(a,b),d)

# ─────────────────────────────────────────────────────────
# SECTOR DEFINITIONS
# ─────────────────────────────────────────────────────────

STRONG_COMPANIES = [
    ("AgriTech",    "IoT & Smart Farming",   ["Idea","Early Traction","Growth"],
     ["UzhavanTech Solutions","Vayal AgriSense","Nattu IoT Labs","Pachai AgroSystems","Selva SmartFarm"],
     ["Erode","Coimbatore","Salem","Thanjavur","Tiruppur"],
     [
         "Solar-powered LoRaWAN sensor nodes monitoring soil NPK, moisture and pH for {crop} farms. "
         "{n} units deployed across {city1} and {city2}. {saving}% water savings and {yield_imp}% yield improvement validated. "
         "Hardware at ₹{hw_price}/unit + ₹{sub}/month farm advisory subscription. TAM: {tam}M irrigated smallholder farms. "
         "Team: {team}.",

         "Drone-based multispectral NDVI crop health surveillance for {crop} clusters. "
         "AI detects pest stress 10 days earlier than visual inspection. "
         "MoU with {n} FPOs covering {area} hectares. ₹{hw_price}/acre/season drone-as-a-service model. "
         "Team: {team}.",

         "AI-powered supply chain linkage platform connecting {crop} smallholder FPOs directly to food processors, "
         "eliminating {n} middlemen layers and increasing farmer netback by {saving}%. "
         "{orders} FPO members onboarded. Revenue: {sub}% platform commission on facilitated trade. "
         "Team: {team}.",
     ],
     [REC_PROCEED, REC_MONITOR]),

    ("HealthTech",  "AI Diagnostics",        ["Prototype","Clinical Validation","Early Traction"],
     ["CardioSense AI","RetinaScan Diagnostics","PulmoAI Labs","DermaVision Health","NeuroBridge Tech"],
     ["Chennai","Coimbatore","Vellore","Madurai"],
     [
         "AI fundus camera analysis for diabetic retinopathy screening at primary health centres. "
         "{n}K clinical images training dataset. {acc}% sensitivity, {spec}% specificity — IRB-validated at {city1} Govt Medical College. "
         "₹{scan}/scan service to government health centres under NHM. "
         "Partners: {n2} district hospitals. Team: {team}.",

         "Portable ECG patch + smartphone AI for atrial fibrillation detection for rural ANMs and ASHA workers. "
         "CDSCO Class B medical device registration submitted. "
         "Pilot with {n} patients across {city1} district, {acc}% clinical accuracy. ₹{sub}/month facility subscription. Team: {team}.",

         "AI skin lesion triage app (melanoma, psoriasis, fungal) validated on {n}K dermoscopy images. "
         "Partnership with {n2} private dermatology chains for white-label deployment. "
         "₹{scan}/consult revenue model. ICMR retrospective study endorsement. Team: {team}.",
     ],
     [REC_PROCEED, REC_MONITOR]),

    ("EdTech",      "Adaptive Learning",     ["Prototype","Early Traction","Growth"],
     ["Kalvi AI Labs","TamilLearn Platform","Padippu Systems","Arivu EdTech","Vidya Tamil Hub"],
     ["Chennai","Coimbatore","Madurai","Salem","Vellore"],
     [
         "Tamil-medium adaptive learning app for Class 6-10 government school students preparing for TN State Board. "
         "AI personalises {content} using spaced repetition. {n} active paid students, {imp}% score improvement in mock assessments. "
         "₹{sub}/month subscription. MoU with {n2} government schools. Team: {team}.",

         "TNPSC Group II/IV exam prep platform in Tamil with vernacular audio-visual explanations. "
         "{n} paid subscribers, {r}% 30-day retention, {n2} daily mock tests attempted. "
         "₹{sub}/month premium plan. Team: {team}.",

         "Gamified STEM skill-building app for polytechnic diploma students aligned to industry 4.0 skill requirements. "
         "IBM-endorsed curriculum. Pilot with {n2} polytechnic colleges. {imp}% placement rate improvement. "
         "₹{sub}/student/year B2B model. Team: {team}.",
     ],
     [REC_PROCEED, REC_MONITOR]),

    ("FinTech",     "Lending & Credit",      ["Early Traction","Growth"],
     ["KanaFin Tech","VelCapital Digital","MaranPay Solutions","ArasuCredit","ThiruFinance"],
     ["Chennai","Coimbatore","Madurai","Tiruppur","Salem"],
     [
         "GST-data and alternate credit scoring platform for textile MSME working capital loans in Tiruppur. "
         "₹{amt}Cr disbursed, {n} borrowers, {npa}% NPA, 48-hour disbursement. "
         "Operating under NBFC co-lending model with {n2} NBFC partners. Revenue: {sub}% net interest margin. Team: {team}.",

         "UPI-based micro-lending for auto-rickshaw and cab drivers collateral-free. "
         "{n} active loan accounts, ₹{amt}Cr AUM, {r}% digital repayment compliance rate. "
         "Partnered with {n2} fleet operators. Revenue: 2%/month interest + 1% processing fee. Team: {team}.",

         "Sachet parametric micro-insurance via WhatsApp for women SHG members in Tamil Nadu. "
         "{n}K policies issued, {n2} SHGs partnered with NABARD. "
         "₹{sub} annual premium, 72-hour claims settlement. Revenue: 15% commission from insurance partner. Team: {team}.",
     ],
     [REC_PROCEED, REC_MONITOR]),

    ("CleanTech",   "EV & Battery Tech",     ["Early Prototype","Prototype","Pilot"],
     ["VoltGrid Systems","NilaEnergy Labs","TerraWatt Innovations","EcoPower Dynamics","SunForge Tech"],
     ["Chennai","Hosur","Coimbatore","Tiruchirappalli"],
     [
         "Micro-channel liquid cooling plates for Li-ion 2W/3W EV packs in 45°C+ tropical conditions. "
         "Patent filed (No. 202541{pat}). 12°C peak cell temp reduction, 40% cycle life extension. "
         "Testing at IIT Madras Research Park. LOI from {n} EV OEMs. "
         "₹{hw_price}L seed ask under TANSEED for ARAI certification. Team: {team}.",

         "Second-life Li-ion battery repurposing system for EV fleet operators — refurbish and redeploy for solar storage. "
         "{saving}% lower cost vs new batteries, {yield_imp}% recovered capacity. "
         "Pilot with {n} EV fleet yards. Revenue: per-kWh energy savings sharing. Team: {team}.",

         "AI solar farm yield optimisation SaaS — soiling loss prediction and tracker fault analytics. "
         "{imp}% AEP improvement validated across {n}MW solar capacity. "
         "₹{sub}L/MW/year subscription. {n2} solar park operator customers. Team: {team}.",
     ],
     [REC_PROCEED, REC_MONITOR]),

    ("SaaS",        "MSME ERP",              ["Prototype","Early Traction","Growth"],
     ["LedgerCloud","FlowSuite Works","OpusHive AI","NexusStack One","OrbitMSME"],
     ["Chennai","Coimbatore","Tiruppur","Hosur"],
     [
         "Integrated ERP and GST compliance SaaS for textile manufacturing MSMEs in Tiruppur. "
         "Reduces manual ledger and GST filing effort by {saving}%. "
         "{n} paying customers, ₹{amt}L ARR, {r}% monthly churn. "
         "₹{sub}/month SME plan. TallyPrime data import ready. Team: {team}.",

         "AI inventory and procurement automation for automotive component suppliers in Chennai-Hosur. "
         "Reduces inventory carrying cost by {saving}% via ML demand forecasting. "
         "POC with {n} Tier-2 suppliers, {r}% converting to paid. Team: {team}.",

         "Workforce attendance, payroll and PF/ESI compliance SaaS for garment exporters in Tiruppur. "
         "{saving}% reduction in payroll processing time. {n} companies, {n2}K employees managed. "
         "₹{sub}/employee/month. EPFO API integrated. Team: {team}.",
     ],
     [REC_PROCEED, REC_MONITOR]),

    ("LogisticsTech","Freight Tech",         ["Early Traction","Growth"],
     ["VelLogistics AI","ThiruNetworks","NammaLink Tech","MetroFreight AI","BayConnect"],
     ["Chennai","Coimbatore","Tiruppur","Madurai"],
     [
         "Full-truckload AI load-matching and dynamic pricing for SME shippers in Tamil Nadu. "
         "{n} trips, ₹{amt}Cr GMV, {saving}% fleet utilisation improvement. "
         "7% platform commission. {n2} active transporters on platform. Team: {team}.",

         "IoT cold-chain logistics platform for perishable agri-produce from farm to retail. "
         "Temperature loggers + predictive spoilage alerts reduced wastage from {saving}% to {yield_imp}%. "
         "{n} FPO shipments/month. Revenue: ₹{sub}/consignment. Team: {team}.",

         "EV micro-mobility last-mile freight delivery platform for e-commerce and kirana resupply in {city1}. "
         "{n} EV bikes, {orders} deliveries/day, {saving}% lower delivery cost vs diesel. "
         "₹{sub}/delivery commission. Partnered with {n2} e-commerce sellers. Team: {team}.",
     ],
     [REC_PROCEED, REC_MONITOR]),

    ("DeepTech",    "Advanced Materials",    ["Early Prototype","Prototype","Pilot"],
     ["NanoForge Systems","QuantumDyne Labs","PlasmaCore Innovations","ApexMeta Works","PrismTech Dynamics"],
     ["Chennai","Hosur","Tiruchirappalli","Coimbatore"],
     [
         "Graphene-enhanced conductive inks for flexible PCBs — 5× conductivity vs silver paste at 60% lower cost. "
         "Patent filed (No. 202541{pat}). Validated at IIT Madras nanofab. "
         "LOI from {n} FPCB manufacturers. ₹{hw_price}L seed ask for scale-up. Team: {team}.",

         "Compact solid-state LiDAR for 2W ADAS at ₹{hw_price}/unit vs ₹80K imports — 100m range, 0.1° angular resolution. "
         "Patent pending. Prototype tested at IIT Madras IITM Research Park. "
         "LOI from {n} OEM development teams. Team: {team}.",

         "High-entropy alloy tooling for precision die casting — {saving}% longer tool life vs H13 steel, {imp}% tighter tolerance. "
         "Validated at TNPL foundry, Tirunelveli. {n} foundry customers in pilot. Team: {team}.",
     ],
     [REC_PROCEED, REC_MONITOR]),

    ("WaterTech",   "Wastewater Treatment",  ["Prototype","Pilot","Early Traction"],
     ["NeerTech","ThanniFlo","AquaClear Labs","PureWave Systems","BlueStream Works"],
     ["Tiruppur","Karur","Chennai","Coimbatore","Madurai"],
     [
         "Electrocoagulation effluent treatment system for textile dyeing MSMEs in Tiruppur. "
         "{saving}% COD reduction and 99% colour removal. TNPCB compliant. "
         "Pilot at {n} dyeing units. BOOT model: ₹{sub}/kL treated. Team: {team}.",

         "Solar-powered ceramic filtration water ATM for fluoride/nitrate-contaminated groundwater in rural TN. "
         "TDS <200 ppm, fluoride <0.5 ppm validated. {n} village panchayats contracted. "
         "₹{sub}/litre revenue. Team: {team}.",

         "AI sewer overflow prediction system for urban ULBs — LSTM on IoT sensor data gives 6-hour advance warning. "
         "{saving}% accurate prediction in {city1} Municipal pilot. "
         "₹{sub}L/ULB/year SaaS. {n2} ULBs in pipeline. Team: {team}.",
     ],
     [REC_PROCEED, REC_MONITOR]),

    ("SpaceTech",   "Drones & UAVs",         ["Early Prototype","Prototype","Pilot"],
     ["SkyAero Systems","VegaDynamics","FalconRobotics","HawkTech Labs","EagleAero India"],
     ["Chennai","Coimbatore","Hosur","Tiruchirappalli"],
     [
         "Fixed-wing hybrid-VTOL agricultural surveillance UAV — 2h endurance, 15km range, 4K multispectral NDVI payload. "
         "DGCA BVLOS waiver application submitted. Pilot with {n} FPOs across {city1} district. "
         "₹{sub}/hectare/survey service model. Team: {team}.",

         "Swarm drone coordination for precision seeding and pesticide application — {n}-drone swarm, <1m GPS accuracy. "
         "Patent pending. Reducing chemical input by {saving}%. Pilot with {n2} plantation companies. Team: {team}.",

         "Compact LEO satellite ground station — X-band tracking, {n}00 Mbps data throughput, ISRO-compatible. "
         "LOI from {n2} NewSpace operators. ₹{sub}Cr/year per station O&M contract. Team: {team}.",
     ],
     [REC_PROCEED, REC_MONITOR]),

    ("RetailTech",  "D2C & eCommerce",       ["Early Traction","Growth"],
     ["KadaiBazaar","NammaThamizh Commerce","SelaiDirect","PattaiMarket","VettiShop"],
     ["Chennai","Madurai","Coimbatore","Kanchipuram"],
     [
         "D2C Kanjivaram and Chettinad handloom textiles platform connecting weavers to urban buyers. "
         "{n} weaver families onboarded, ₹{amt}L GMV in {months} months, {r}% repeat purchase. "
         "35% gross margin. Instagram live commerce + D2C website. Team: {team}.",

         "GI-tagged natural dye block-print fabric brand sourcing from women SHG cooperatives in Nagapattinam. "
         "{n} artisan producers, {orders} orders/month, ₹{sub} AOV. "
         "B2B corporate gifting + D2C subscription box. Team: {team}.",
     ],
     [REC_PROCEED, REC_MONITOR, REC_MORE_INFO]),
]

WEAK_COMPANIES = [
    ("HealthTech",  "REJECT_HIGH_RISK",
     [
         "MiracleCure Health claims a wearable frequency bracelet that 'cures cancer in 30 days using quantum healing vibrations'. "
         "Tested on {n} family members — all recovered. Seeking ₹{amt}Cr to scale production. No clinical data. Team: {team}.",

         "WonderWellness sells an alkaline ionizer promising to 'reverse diabetes permanently'. "
         "98% success rate from customer testimonials. Founder has a certificate in alternative medicine. "
         "Requesting ₹{amt}Cr grant for 'marketing and office space'. No CDSCO clearance plan. Team: {team}.",
     ],
     REC_REJECT),

    ("EdTech",      "REJECT_HIGH_RISK",
     [
         "FutureEduStar wants to build 'the next BYJU's but better'. "
         "Will have all subjects Class 1-12, target {amt} Crore users in 2 years. "
         "Needs ₹{amt2}Cr to hire {n} teachers. Founder 'watched many EdTech tutorials'. No product built. Team: {team}.",
     ],
     REC_REJECT),

    ("SaaS",        "REQUEST_MORE_INFORMATION",
     [
         "SmartAgriHub wants to build an app for farmers in Tamil Nadu. "
         "Will decide features after launch based on what users say. "
         "Needs ₹{amt}L to pay developer salaries. Team: {team}.",

         "GlobalEduWorld plans a software for textile MSMEs. "
         "No tech stack, features, or pricing defined yet. "
         "Requesting ₹{amt}L from StartupTN. Mentions serving both farmers and factory owners interchangeably. Team: {team}.",
     ],
     REC_MORE_INFO),

    ("FinTech",     "REJECT_HIGH_RISK",
     [
         "CryptoWealthTN promises 40% monthly returns via 'AI crypto arbitrage'. "
         "Seeking ₹{amt}Cr from StartupTN as seed capital. No RBI registration, no regulatory framework addressed. Team: {team}.",
     ],
     REC_REJECT),

    ("AgriTech",    "REQUEST_MORE_INFORMATION",
     [
         "GreenAgriConnect will build a marketplace for farmers. No unique value proposition stated. "
         "Team of {n} freshers. Seeking ₹{amt}L. No differentiation from existing platforms like AgriBazaar or DeHaat. Team: {team}.",
     ],
     REC_MORE_INFO),

    ("CleanTech",   "REQUEST_MORE_INFORMATION",
     [
         "EcoGreenPower plans to install solar panels across Tamil Nadu villages. "
         "No DISCOM approvals, no land access plan, no technical team. "
         "Seeking ₹{amt}Cr with 'details to be worked out later'. Team: {team}.",
     ],
     REC_MORE_INFO),
]

TEAMS = [
    "2 IIT Madras alumni (ML + hardware) and an ex-{company} domain expert",
    "TNAU agronomy graduate and embedded systems engineer with {n} years experience",
    "ex-{company} product manager and a NIT Trichy CS graduate",
    "Ex-{company} domain expert, chartered accountant, and a full-stack developer",
    "IIT Madras PhD researcher and an industry professional with {n} years sector experience",
    "ex-DRDO scientist and 2 IIT Madras engineering graduates",
    "MBBS doctor and 2 computer vision engineers from IIT Madras",
    "Anna University alumni team with prior startup experience",
    "Ex-{company} operations lead and a data scientist",
    "Government hospital clinician and a software architect",
    "Founder with {n} years field experience and a technical co-founder",
]

CORPORATES = ["Mahindra","TVS","HDFC Bank","Zoho","Freshworks","TCS","Infosys","Wipro","HCL","BSNL","TNEB"]

def make_team():
    t = _pick(TEAMS)
    t = t.replace("{company}", _pick(CORPORATES))
    t = t.replace("{n}", str(_ri(3,12)))
    return t

def make_strong_example(idx: int) -> dict:
    row = _pick(STRONG_COMPANIES)
    industry, category, stages, companies, cities, proposals, recs = row

    company = _pick(companies)
    city1 = _pick(cities)
    city2 = _pick([c for c in cities if c != city1] or cities)
    stage = _pick(stages)
    rec = _pick(recs)

    proposal_tmpl = _pick(proposals)

    crops = ["paddy","sugarcane","banana","turmeric","coconut","groundnut","cotton","mango"]
    contents = ["concept videos and Tamil notes","MCQ drill banks and live doubt sessions","gamified micro-lessons"]

    proposal = proposal_tmpl.format(
        crop=_pick(crops),
        city1=city1, city2=city2,
        n=_ri(10,500), n2=_ri(2,20),
        saving=_ri(15,55), yield_imp=_ri(10,40),
        imp=_ri(10,40),
        hw_price=_ri(8,200),
        sub=_ri(100,5000),
        sub2=_ri(5000,50000),
        tam=round(random.uniform(0.3,3.5),1),
        area=_ri(200,8000),
        orders=_ri(50,5000),
        amt=round(random.uniform(0.5,30.0),1),
        amt2=_ri(5,50),
        npa=round(random.uniform(0.3,4.0),1),
        r=_ri(50,95),
        acc=_ri(87,97),
        spec=_ri(86,96),
        scan=_ri(30,500),
        months=_ri(3,18),
        pat=_ri(100000,999999),
        content=_pick(contents),
        team=make_team(),
    )

    return _build_example(idx, industry, category, stage, company, city1, proposal, rec)

def make_weak_example(idx: int) -> dict:
    row = _pick(WEAK_COMPANIES)
    industry, _, proposals, rec = row

    company = f"{_pick(['Smart','Digital','Future','Wonder','Miracle'])}{industry.replace('Tech','')}TN"
    city = _pick(TN_CITIES)
    stage = "Idea"

    proposal_tmpl = _pick(proposals)
    proposal = proposal_tmpl.format(
        n=_ri(3,10),
        amt=_ri(1,5),
        amt2=_ri(5,50),
        team=make_team(),
    )

    return _build_example(idx, industry, "Early Stage", stage, company, city, proposal, rec)

def _build_example(idx, industry, category, stage, company, city, proposal, rec):
    conf = {
        REC_PROCEED:   _rf(0.82, 0.96),
        REC_MONITOR:   _rf(0.76, 0.90),
        REC_MORE_INFO: _rf(0.78, 0.92),
        REC_REJECT:    _rf(0.88, 0.98),
    }[rec]

    strong_strengths = [
        f"Experienced team with deep domain expertise in {industry}",
        "Clear problem-solution fit validated with real customers or pilots",
        "Defensible IP, patent, or proprietary technology advantage",
        "Strong alignment with Tamil Nadu government priority sectors and schemes",
        "Realistic and detailed financial projections backed by traction evidence",
        "Vernacular Tamil-first market adaptation for regional accessibility",
        "Well-articulated go-to-market strategy with credible channel partnerships",
        "Scalable recurring revenue model with demonstrated customer retention",
    ]
    weak_strengths = [
        "Addresses a genuine public-interest problem in Tamil Nadu",
        "Team demonstrates passion and commitment to the domain",
        "Identified a real market gap even if solution is underdeveloped",
    ]

    proceed_weaknesses = [
        "Long B2B sales cycle may slow initial revenue ramp-up",
        "Dependent on regulatory approval timelines outside team control",
        "Hardware-heavy model requires significant working capital for inventory",
        "Limited marketing budget may constrain early-stage customer acquisition",
    ]
    reject_weaknesses = [
        "Completely lacks a validated business model or realistic revenue projection",
        "No technical proof-of-concept, prototype, or pilot data presented",
        "Unrealistic performance or impact claims without supporting evidence",
        "Inappropriate allocation of requested funds (non-R&D expenses)",
        "Regulatory compliance requirements entirely unaddressed in the proposal",
        "No competitive differentiation from well-established existing solutions",
    ]
    info_weaknesses = [
        "Insufficient technical detail to independently assess solution feasibility",
        "Vague financial projections without documented assumptions",
        "Target customer persona is inconsistent or poorly defined",
        "No prototype, pilot, or user validation evidence provided",
    ]

    proceed_risks = [
        "Market adoption risk if customer education investment is underestimated",
        "Competitive entry risk from well-funded incumbents",
        "Regulatory changes in target sector may affect operating model",
        "Key-person dependency risk given current team structure",
    ]
    reject_risks = [
        "High risk of financial misuse given absence of credible project plan",
        "Severe regulatory compliance violations could lead to legal liability",
        "Unvalidated technical claims pose significant product failure probability",
        "Consumer safety and ethical liability risk from unproven technology",
    ]

    if rec == REC_PROCEED:
        strengths = random.sample(strong_strengths, _ri(3,5))
        weaknesses = random.sample(proceed_weaknesses, _ri(2,3))
        risks = random.sample(proceed_risks, _ri(2,3))
        scalability = f"High scalability potential across other Indian states following Tamil Nadu market validation, aligned with national {industry} policies."
    elif rec == REC_MONITOR:
        strengths = random.sample(strong_strengths, _ri(2,4))
        weaknesses = random.sample(proceed_weaknesses + info_weaknesses, _ri(2,3))
        risks = random.sample(proceed_risks, _ri(2,3))
        scalability = f"Moderate scalability potential, contingent on successful pilot completion and resolution of identified operational and regulatory dependencies."
    elif rec == REC_REJECT:
        strengths = random.sample(weak_strengths, _ri(1,2))
        weaknesses = random.sample(reject_weaknesses, _ri(3,5))
        risks = random.sample(reject_risks, _ri(2,4))
        scalability = "No credible scalability pathway identified given the absence of a validated business model or technical proof of concept."
    else:
        strengths = random.sample(weak_strengths, _ri(1,2))
        weaknesses = random.sample(info_weaknesses, _ri(2,3))
        risks = random.sample(proceed_risks[:2] + ["Insufficient information to fully assess technical or regulatory risk exposure."], _ri(2,3))
        scalability = f"Scalability cannot be assessed without additional technical and market validation data from the team."

    summaries = [
        f"{company} proposes a {industry.lower()} solution targeting a validated market gap in Tamil Nadu at {stage.lower()} stage.",
        f"{company} is developing a {industry.lower()} product addressing a real pain point in the Tamil Nadu market.",
        f"{company} presents a {stage.lower()}-stage {industry.lower()} initiative with clear regional focus on Tamil Nadu priority sectors.",
    ]
    problems = [
        f"Existing market inefficiencies and absence of technology-driven solutions in the {industry.lower()} sector in Tamil Nadu.",
        f"High operational costs, information asymmetry, and poor access to quality services for the target segment.",
        f"Under-served demand in Tamil Nadu's {industry.lower()} sector due to absence of affordable and localised technology solutions.",
    ]
    solutions = [
        f"A technology-driven platform addressing the identified gap through AI, IoT, and deep domain knowledge.",
        f"Purpose-built software and hardware solution integrating sector expertise with modern engineering.",
        f"Data-driven decision-support system leveraging proprietary technology adapted to Tamil Nadu conditions.",
    ]
    target_markets = [
        f"Tamil Nadu {industry.lower()} sector operators and small businesses, with a large addressable underserved market.",
        f"Government, institutional buyers, and private enterprises in Tamil Nadu's growing {industry.lower()} ecosystem.",
        f"SMEs and consumers across Tamil Nadu cities and districts in the {industry.lower()} value chain.",
    ]
    biz_models = [
        "B2B SaaS and platform model with recurring subscription revenue and enterprise tier pricing.",
        "Hardware + software bundle with recurring service and maintenance revenue streams.",
        "Marketplace commission model supplemented by premium analytics and advisory services.",
        "B2G and B2B service delivery with long-term government and institutional contracts.",
    ]
    rev_models = [
        "Monthly subscription fees with tiered pricing based on usage, user count, or transaction volume.",
        "Hardware sale margin combined with recurring annual maintenance and software licensing fees.",
        "Transaction commission (5-15%) on facilitated value plus optional premium feature subscriptions.",
        "Per-unit or per-outcome pricing aligned with customer value realisation and success metrics.",
    ]

    evaluation = {
        "summary":        _pick(summaries),
        "problem":        _pick(problems),
        "solution":       _pick(solutions),
        "target_market":  _pick(target_markets),
        "business_model": _pick(biz_models),
        "revenue_model":  _pick(rev_models),
        "strengths":      strengths,
        "weaknesses":     weaknesses,
        "risks":          risks,
        "scalability":    scalability,
        "recommendation": rec,
        "confidence_score": conf,
    }

    user_content = (
        f"Metadata: Industry: {industry}, Category: {category}, Stage: {stage}\n\n"
        f"Please evaluate the following StartupTN startup proposal:\n\n"
        f"Company Name: {company}\n"
        f"Location: {city}, Tamil Nadu\n"
        f"Proposal: {proposal}"
    )

    return {
        "proposal_id": f"PROP-TN-SYN-{idx:04d}",
        "industry": industry,
        "category": category,
        "is_synthetic": True,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
            {"role": "assistant", "content": json.dumps(evaluation, ensure_ascii=False)},
        ],
    }

def generate_dataset(count: int, output_path: str, seed: int = 42):
    random.seed(seed)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("  StartupTN Synthetic Dataset Generator")
    print(f"  Target examples : {count}")
    print(f"  Output          : {output_path}")
    print(f"  Random seed     : {seed}")
    print("="*60)

    examples = []
    rec_counts: Dict[str,int] = {}
    industry_counts: Dict[str,int] = {}

    # ~85% strong (approvable/monitor), ~15% weak (reject/info)
    weak_indices = set(random.sample(range(count), int(count * 0.15)))

    for i in range(count):
        try:
            if i in weak_indices:
                ex = make_weak_example(i+1)
            else:
                ex = make_strong_example(i+1)
            # Sanity-check: assistant content must be valid JSON
            ev = json.loads(ex["messages"][2]["content"])
            assert "recommendation" in ev and "confidence_score" in ev
            examples.append(ex)
            rec_counts[ev["recommendation"]] = rec_counts.get(ev["recommendation"], 0) + 1
            industry_counts[ex["industry"]] = industry_counts.get(ex["industry"], 0) + 1
        except Exception as e:
            print(f"  [SKIP] idx {i+1}: {e}")

    with open(output_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    total = len(examples)
    print(f"\n  ✅ Generated {total} valid examples\n")
    print("  Recommendation distribution:")
    for rec, cnt in sorted(rec_counts.items(), key=lambda x: -x[1]):
        pct = round(cnt/total*100, 1)
        bar = "█" * (cnt // max(1, total//50))
        print(f"    {rec:<44} {cnt:>4} ({pct:>5.1f}%) {bar}")
    print("\n  Industry distribution:")
    for ind, cnt in sorted(industry_counts.items(), key=lambda x: -x[1]):
        print(f"    {ind:<30} {cnt:>4}")
    print(f"\n  Written to: {output_path}")
    print("="*60)
    return examples

def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic StartupTN proposal training data")
    parser.add_argument("--count",  type=int, default=600, help="Number of examples (default: 600)")
    parser.add_argument("--output", type=str, default="data/processed/startup_proposals_synthetic.jsonl")
    parser.add_argument("--seed",   type=int, default=42)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    generate_dataset(count=args.count, output_path=args.output, seed=args.seed)
