# Security Incident Response Policy

## Scope and Definitions

This policy governs the detection, classification, and remediation of security
incidents affecting ACME Robotics systems, customer data, or production
infrastructure.

A security *event* is any observable occurrence in a system or network. A
security *incident* is an event that has been assessed as having an actual or
potential adverse effect on confidentiality, integrity, or availability. All
incidents are events; most events are not incidents.

## Severity Classification

Incidents are classified into four severity levels at the point of triage.

**SEV-1 — Critical.** Confirmed unauthorised access to customer data, active
exploitation of a production system, or a complete outage of a customer-facing
service. Requires immediate escalation to the on-call security lead and the CTO.
Acknowledgement is required within fifteen minutes, at any hour.

**SEV-2 — High.** Suspected unauthorised access without confirmation, a
vulnerability with a published exploit affecting production, or degraded
availability of a customer-facing service. Acknowledgement is required within
one hour during business hours and within four hours outside them.

**SEV-3 — Moderate.** A vulnerability with no known public exploit, a security
misconfiguration in a non-production environment, or a policy violation with no
evidence of data exposure. Acknowledgement is required within one business day.

**SEV-4 — Low.** Informational findings, hardening opportunities, and issues
with no plausible path to exploitation. Triaged during the weekly security
review; no acknowledgement deadline applies.

## Escalation Path

The first responder is always the engineer on the security on-call rotation,
reachable through the `#sec-oncall` channel and the paging system. For SEV-1
incidents the on-call engineer must page the security lead directly rather than
relying on channel notification.

If the security lead does not acknowledge a SEV-1 page within ten minutes, the
on-call engineer escalates to the CTO. If the CTO does not acknowledge within a
further ten minutes, the on-call engineer escalates to the CEO. Escalation is
never considered an overreaction, and no employee will face any negative
consequence for escalating an incident that is later downgraded.

## Containment and Evidence

Containment takes priority over root-cause analysis. Responders should isolate
affected systems before beginning forensic work.

Before terminating or rebuilding any compromised host, capture a memory image
and a disk snapshot. Terminating a compromised instance without capturing
evidence destroys the only reliable record of attacker behaviour and is treated
as a process failure in the post-incident review.

Credentials known or suspected to be exposed must be rotated within one hour of
discovery for SEV-1 and SEV-2 incidents. Rotation includes revoking active
sessions, not merely changing the stored secret.

## Customer Notification

Where an incident involves confirmed unauthorised access to customer data,
affected customers are notified within seventy-two hours of confirmation. The
notification is drafted by the security lead, reviewed by Legal, and sent by the
account team.

Notifications state what data was affected, the window during which the access
occurred, the remediation performed, and the action the customer should take.
Speculation about attacker identity or motive is excluded from all customer
communications.

## Post-Incident Review

Every SEV-1 and SEV-2 incident requires a written post-incident review, published
within ten business days of resolution. Reviews are blameless: they describe
system and process failures, never individual fault.

Each review must produce at least one tracked remediation action with a named
owner and a due date. Remediation actions from SEV-1 reviews are reviewed
monthly by the engineering leadership team until closed.
