#include <threes_Mechanics.h>

int main() {
  Direction d = D;
  Direction u = U;

  Shift s0(d, 1);
  Shift s1(d, 0);
  std::vector<Shift> shifts;
  s0.string_vec.push_back(0);
  s0.string_vec.push_back(1);
  s0.string_vec.push_back(0);
  s0.string_vec.push_back(3);

  s1.string_vec.push_back(0);
  s1.string_vec.push_back(1);
  s1.string_vec.push_back(0);
  s1.string_vec.push_back(3);


  Node n1;
  Node n2;
  Node n3;
  Node n4;


  n1.f = 990;
  n2.f = 104;
  n3.f = 50;
  n4.f = 1000;

  NodeQ nq;
  nq.push(n4);
  nq.push(n2);
  nq.push(n1);
  nq.push(n3);

  // should print 1000, 990, 104, 50
  while (!nq.empty()) {
    std::cout << nq.top().f << "\n";
    nq.pop();
  }

  if (s0 < s1) {
    std::cout << "s0 < s1\n";
  } else {
    std::cout << "s0 > s1\n";
  }

  shifts.push_back(s0);
  shifts.push_back(s1);
  std::sort(shifts.begin(), shifts.end());

  for (Shift s : shifts) {
    std::cout << "id: " << s.id << ", str: ";
    for (int i = 0; i < 4; i++) {
      std::cout << s.string_vec[i] << " ";
    }
    std::cout << "\n";
  }

  
}