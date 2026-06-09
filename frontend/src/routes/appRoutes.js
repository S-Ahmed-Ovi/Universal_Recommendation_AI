import { lazy } from "react";

const Dashboard = lazy(() => import("../pages/Dashboard"));
const Datasets = lazy(() => import("../pages/Datasets"));
const Training = lazy(() => import("../pages/Training"));
const ResultsHub = lazy(() => import("../pages/ResultsHub"));
const Benchmarks = lazy(() => import("../pages/Benchmarks"));
const ModelComparison = lazy(() => import("../pages/ModelComparison"));
const DeploymentCenter = lazy(() => import("../pages/DeploymentCenter"));
const Registry = lazy(() => import("../pages/Registry"));
const Integrations = lazy(() => import("../pages/Integrations"));
const Analytics = lazy(() => import("../pages/Analytics"));
const APIPlayground = lazy(() => import("../pages/APIPlayground"));
const Tenants = lazy(() => import("../pages/Tenants"));
const Notifications = lazy(() => import("../pages/Notifications"));
const Settings = lazy(() => import("../pages/Settings"));
const Scheduler = lazy(() => import("../pages/Scheduler"));

export const protectedRoutes = [
  { path: "/dashboard", component: Dashboard },
  { path: "/datasets", component: Datasets },
  { path: "/training", component: Training },
  { path: "/results", component: ResultsHub },
  { path: "/benchmarks", component: Benchmarks },
  { path: "/model-comparison", component: ModelComparison },
  { path: "/deployment", component: DeploymentCenter },
  { path: "/registry", component: Registry },
  { path: "/integrations", component: Integrations },
  { path: "/analytics", component: Analytics },
  { path: "/api", component: APIPlayground },
  { path: "/tenants", component: Tenants },
  { path: "/notifications", component: Notifications },
  { path: "/settings", component: Settings },
  { path: "/scheduler", component: Scheduler },
];
