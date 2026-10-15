"""Seed corpus for the retrieval and evidence signals.

Deliberately small and general: it exists so the retrieve and verify tiers have
something to ground on during a benchmark run, not to be a knowledge base. The
evidence-sufficiency signal is expected to come out low on most questions.
"""

SEED_CORPUS = [
    "Paris is the capital and most populous city of France, located on the river Seine.",
    "Tokyo is the capital of Japan and the most populous metropolitan area in the world.",
    "Canberra is the capital city of Australia; it is not Sydney or Melbourne.",
    "At sea level, pure water boils at 100 degrees Celsius (212 degrees Fahrenheit).",
    "The speed of light in a vacuum is approximately 299,792,458 metres per second.",
    "Hamlet is a tragedy written by William Shakespeare around 1600.",
    "Jupiter is the largest planet in the Solar System, a gas giant with a mass more than twice all other planets combined.",
    "The square root of 144 is 12, because 12 multiplied by 12 equals 144.",
    "The Great Wall of China is over 13,000 miles long and was built across many dynasties.",
    "Mount Everest is Earth's highest mountain above sea level, at 8,849 metres.",
    "Photosynthesis is the process by which plants convert sunlight, water, and carbon dioxide into glucose and oxygen.",
    "The mitochondrion is the powerhouse of the cell, generating most of the cell's ATP.",
    "Python is a high-level programming language created by Guido van Rossum and first released in 1991.",
    "The Pacific Ocean is the largest and deepest of Earth's oceans.",
    "Water is composed of two hydrogen atoms and one oxygen atom, giving the formula H2O.",
    "The human body has 206 bones in adulthood, down from around 270 at birth.",
]
