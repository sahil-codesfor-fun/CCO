
/**
 * Cloud Cost Optimizer — Traffic Tracker
 * 
 * Instructions:
 * 1. Host the `ingress.py` API on a reachable server.
 * 2. Add the following script to your website's <head> or <body>.
 * 3. Replace `https://your-ingress-api.com` with your real endpoint.
 */

(function() {
    const INGRESS_API = "http://localhost:8000/track";
    const SESSION_ID = 'sess_' + Math.random().toString(36).substring(2, 9);
    
    function reportHit() {
        console.log("Reporting site traffic to Cloud Cost Optimizer...");
        fetch(INGRESS_API, {
            mode: 'no-cors', // For light-weight heartbeats
            cache: 'no-cache'
        }).catch(err => console.error("CloudTracker: Ingress offline. Skipping hit."));
    }

    // Report hit on page load
    reportHit();

    // Strategy: Also report minor high-energy events (clicks, scrolls)
    // Every few interactions count as a "hit" so we see more detail.
    let interactionCount = 0;
    const INTERACTION_THRESHOLD = 5;

    document.addEventListener('click', () => {
        interactionCount++;
        if (interactionCount >= INTERACTION_THRESHOLD) {
            reportHit();
            interactionCount = 0;
        }
    });

    console.info("Cloud Cost Optimizer: Traffic tracking initialized for session " + SESSION_ID);
})();
