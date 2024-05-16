#ifndef ML_MODEL_H
#define ML_MODEL_H

#include <string>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include "sql/join_optimizer/access_path.h"
#include "sql/join_optimizer/make_join_hypergraph.h"
#include "sql/join_optimizer/relational_expression.h"
#include "sql/handler.h"

/**
  This class represents the MLSocket client used for connecting to a ML
  model.
*/
class MLSocket {
 public:
  MLSocket(const std::string &socket_path);
  ~MLSocket();

  bool ConnectSocket();
  bool WriteML(const std::string &data);
  bool ReadML();
  char* GetData();
  void Disconnect();

 private:
  std::string m_socket_path;
  static const int m_data_size = 1024;
  char m_data[m_data_size];
  int m_socket_fd;
  bool m_is_connected;
};


/**
  This class represents the MLQueryRep,
  which is the query representation the ML model parses as input.
*/
class MLQueryRep {
 public:
  MLQueryRep();
  ~MLQueryRep();

  std::string CreateQueryRep(JoinHypergraph *graph, const JoinPredicate *edge, AccessPath *left_path, AccessPath *right_path, hypergraph::NodeMap right);

 private:
  std::string m_tables;
  std::string m_joins;
  std::string m_predicates;

  std::vector<std::string> m_operators;

  void AddTables(JoinHypergraph *graph, const JoinPredicate *edge);
  void AddJoins(JoinHypergraph *graph, const JoinPredicate *edge, hypergraph::NodeMap right);
  void AddPredicates(JoinHypergraph *graph, const JoinPredicate *edge, AccessPath *left_path, AccessPath *right_path);
  std::string PrepareCondition(const std::string &condition);
  std::string GetCondition(Item *condition);
  void ParsePredicates(JoinHypergraph *graph, const std::string &join_condition, AccessPath *path);
};

/**
  This class represents the MLModel. It communicates through the MLSocket with the actual ML model,
  obtaining its cardinality estimate.
*/
class MLModel {
 public:
  MLModel();
  ~MLModel();

  double GetCardinalityEstimate(JoinHypergraph *graph, const JoinPredicate *edge, AccessPath *left_path, AccessPath *right_path, hypergraph::NodeMap right);

 private:
  MLQueryRep m_ml_query_rep;
  double m_cardinality_estimate;
};

extern MLSocket ml_socket;

#endif  // ML_MODEL_H
