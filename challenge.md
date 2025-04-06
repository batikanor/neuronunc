Track 4: Next Level Coffee – Brewing Better
Coffee with AI
Coffee is personal, complex, and increasingly powered by data. At Next Level Coffee (NLC),
we’ve built nunc. —a connected espresso machine and grinder that captures detailed IoT data
from every single shot brewed and bean ground. In this track, you’ll work with real data from
nunc. to either optimize the brewing process or personalize the coffee experience through AI.
Your solution should show how technology can elevate the art and science of coffee.
 Choose one of two use cases
 Use Case 1: Brew Optimization
Can we predict the perfect grind size before we even grind? Your task:
Build a model or AI agent that uses historical brewing and grinding data along with insights into
the physics of espresso machines to recommend the ideal grind size before a shot is pulled.
The goal? Ensure the machine reaches the desired peak pressure of 6–9 bars during
extraction. This will improve consistency, reduce waste, and elevate every brew.
Your model will receive real-time input such as:
{
"deviceId": "10000000bce3a15f",
"consumableRoastId": 2,
"previousConsumableRoastId": 2,
"consumableId": 20222,
"recipeId": 31,
"recipeGrindSize": 10,
"grinderTemperature": 280,
"dosedWeight": 181,
"ambientTemperature": 200,
"ambientHumidity": 0.59,
"daysConsumableOpened": 3
}
Your model should respond with:
{
"recommendedGrindSize": 117
}
Your solution should be designed for seamless integration into the nunc. ecosystem and should
minimize the number of espresso shots that fall outside the optimal 6–9 bar extraction
pressure range, improving consistency and brew quality over time.
 Use Case 2: Instant Dial-In - Scan Your Coffee, Personalize Your Flavor
Can we dial in the perfect espresso just by looking at the coffee? This use case is especially
designed for coffee nerds. Your task:
Build a model or AI agent that uses visual input (like coffee bag photos, roast level, origin, and
tasting notes) and combines it with knowledge of coffee extraction science to recommend the
ideal espresso shot parameters for a new coffee.
The goal? Enable users to simply scan or upload a photo* of the coffee and instantly receive
personalized dial-in recommendations—no guesswork, no waste.
In addition to image and coffee data, the model should also ask the user for their desired
flavor experience—for example: balanced, fruity, bold, sweet, or chocolatey. Based on both
the realistic potential of the coffee (roast level, processing, origin) and the user’s input, the
model should adjust the recipe accordingly.
Your model should determine the shot based on the following parameters:
• Flow or pressure control of the shot — recommend a target flow rate or pressure profile
• Brewing temperature
• Grind setting, using 120 µm of the “BAR ITALIA” as the reference point
• Brew ratio — the ratio of ground coffee to total water used
Your solution should integrate effortlessly into the nunc. ecosystem, combining visual
recognition, taste profiling, and extraction physics to deliver fast, high-fidelity recipes tailored
to both the coffee and the drinker—all with a single scan and a tap.
*Sample images:
Required: Actionable Insights (25% of Evaluation)
Regardless of which use case you pursue; your team must analyze the dataset provided and
surface at least 3 actionable insights. These insights should help improve either:
• The engineering of the nunc. machine (hardware, sensors, control systems)
• The user brewing experience (consistency, personalization, usability)
These insights will account for 25% of your overall evaluation score. We’re looking for curious
minds who dig deep and bring fresh ideas to the surface.
 Data Assets Provided by NLC
We’re providing real IoT data from nunc. machines to fuel your solution:
1. Grind Data (grind_complete_event)
Includes:
• grindSize, recipeBeanWeight, grinderTemperature, totalDosedWeight
• Timing data: shutterTimes, dosedWeights
• Metadata like consumableId, and consumableRoastId etc.
2. Brew Data (brew_complete_event)
Includes:
• Time-series sensor values over extraction (pressures, temps, humidity, flow, valve states)
• peakPressure, brewDuration, brewHeadTemperature, ambientConditions
 Linking the Data
Use the shared eventId to join grind and brew records.
 Note: Some grind events may not have a corresponding brew, so ensure your solution
handles partial data gracefully
 Handling Device Identity
Because the same prototype can change deviceId during updates, use the provided mapping to
group data correctly:
{
 "18446744072574007284": "10000000812efd14",
 "18446744073136002376": "10000000812efd14",
 "18446744073527372579": "10000000dcdd8d82",
 "18446744072890410450": "10000000dcdd8d82",
 "18446744072773813932": "10000000dcdd8d82",
 "18446744072672130913": "10000000bce3a15f",
 "18446744072802279265": "10000000bce3a15f",
 "18446744072772856228": "100000001bcb62d3",
 "18446744072720600839": "1000000036a2c7bc",
 "18446744072835879765": "1000000036a2c7bc",
 "18446744072818439117": "1000000036a2c7bc"
}
 Evaluation Criteria
Your solution will be evaluated across five key categories. Make sure to balance creativity with
clarity, and insights with implementation.
 Solution
Effectiveness
Does your model or agent deliver accurate, meaningful results?
For Use Case 1, does your grind size lead to 6–9 bar
pressure? For Use Case 2, are the recommendations aligned
with user preferences?
30%
 Actionable
Insights
Have you surfaced at least 3 valuable insights from the data?
Do these insights help improve nunc.’s hardware, software, or
user experience? This is a required part of your submission.
25%
 Technical Depth
& Creativity
Is your solution smartly engineered? Is it novel, or does it
improve upon existing ideas in a meaningful way?
20%
 Integration
Readiness
Could this realistically plug into the nunc. platform? Is it robust
and built with real-world constraints in mind?
15%
 Clarity of
Communication
How well do you communicate your approach? Are your
visuals, write-ups, or demos easy to follow and thoughtfully
designed?
10%
 Track prize – pick your perk
Each member of the winning team in the Next Level Coffee track will get to choose one of the
following exclusive rewards:
 1 Year of Free CoffeeA full year of hand-picked specialty coffee delivered monthly to fuel
your inner barista. (up to 15kg/year)
 1-Year AI Subscription of Your Choice
Supercharge your productivity with the AI tool that fits your workflow best—whether it’s Cursor
Pro, Notion AI, GitHub Copilot, RunwayML, Replit, Ghostwriter, or any other AI
subscription. You choose, we cover it (up to €300/year).
 6 Months of Coffee + 6 Months of AI Tools
Balance your grind—half a year of specialty coffee to fuel your mornings, plus 6 months of your
favorite AI subscription to power your projects.
Plus:
We’re always on the lookout for bright, curious minds to join the team at Next Level Coffee.
This hackathon is also an opportunity to get on our radar for full-time roles in AI, engineering,
and product innovation.